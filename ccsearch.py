#!/usr/bin/env python3
"""ccs: search your Claude Code sessions in natural language.

    ccs "the chat where I found all the connector issues"
    ccs -a "where did I decide to drop the dual-vector embedding idea"   # Claude reranks + explains
    ccs -p my-api --since 2026-10-01 "flaky retry test in the payments client"

Indexes ~/.claude/projects/*/*.jsonl incrementally (only changed files are re-read)
into ~/.cache/ccsearch/index.sqlite. Retrieval is hybrid: BM25 over prompts, replies,
tool-action descriptions and compaction summaries, plus Model2Vec embeddings, fused
with reciprocal rank and rolled up per session. Query typos are corrected against the
corpus vocabulary. Pick a result to resume it in its original directory.
"""

import argparse
import concurrent.futures
import datetime as dt
import difflib
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import orjson

    _loads = orjson.loads
except ImportError:  # pragma: no cover
    _loads = json.loads

PROJECTS = Path(os.environ.get("CCS_PROJECTS", Path.home() / ".claude" / "projects"))
CACHE = Path(os.environ.get("CCS_CACHE", Path.home() / ".cache" / "ccsearch"))
DB_PATH = CACHE / "index.sqlite"
MODEL_NAME = "minishlab/potion-retrieval-32M"
SCHEMA_VERSION = "4"

RRF_K = 30
AGG_W = (1.0, 0.35, 0.15)
PIECE = 700  # chars per embedded chunk; Model2Vec mean-pools, so long chunks go generic
STOP = set(
    "a an and are as at be but by can chat claude did do does done for from had has have how i "
    "if in into is it its me my of on or session so that the their them then there these this "
    "those to was we were what when where which who why will with you your all any find found "
    "conversation one about".split()
)

# --------------------------------------------------------------------------- parsing

TAG_RE = re.compile(
    r"<(system-reminder|local-command-caveat|local-command-stdout|local-command-stderr|"
    r"command-message|bash-stdout|bash-stderr|task-notification)>.*?</\1>",
    re.S,
)
PASTE_RE = re.compile(r"<pasted_content[^>]*>(.*?)</pasted_content>", re.S)
CMD_RE = re.compile(r"<command-name>(.*?)</command-name>", re.S)
ARGS_RE = re.compile(r"<command-args>(.*?)</command-args>", re.S)
BASH_IN_RE = re.compile(r"<bash-input>(.*?)</bash-input>", re.S)
LINK_RE = re.compile(
    r"https://(?:github\.com/[\w.-]+/[\w.-]+/pull/\d+"
    r"|linear\.app/[\w-]+/issue/[A-Z]+-\d+(?:/[\w-]+)?"
    r"|claude\.ai/(?:code/)?artifact/[\w-]+)"
)
GH_TITLE_RE = re.compile(r"gh (?:pr|issue) create[^\n]*?--title\s+([\"'])(.+?)\1")
CONTINUED = "This session is being continued from a previous conversation"


def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
        )
    return ""


def clean_prompt(s):
    """User prompt text as the user typed it, minus harness wrappers."""
    m = CMD_RE.search(s)
    if m:
        a = ARGS_RE.search(s)
        return f"{m.group(1).strip()} {a.group(1).strip() if a else ''}".strip()
    b = BASH_IN_RE.search(s)
    if b:
        return "! " + b.group(1).strip()
    s = TAG_RE.sub("", s)
    s = PASTE_RE.sub(lambda m: " [pasted] " + " ".join(m.group(1).split())[:500] + " ", s)
    return " ".join(s.split())


def pieces(text, limit=PIECE, max_pieces=4):
    """Split text on paragraph/sentence boundaries into <=limit-char pieces."""
    text = text.strip()
    if not text:
        return []
    out, cur = [], ""
    for para in re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z])", text):
        para = " ".join(para.split())
        while len(para) > limit:
            if cur:
                out.append(cur)
                cur = ""
            out.append(para[:limit])
            para = para[limit:]
        if len(cur) + len(para) + 1 > limit and cur:
            out.append(cur)
            cur = para
        else:
            cur = f"{cur} {para}".strip()
    if cur:
        out.append(cur)
    if len(out) > max_pieces:  # keep the opening and the conclusion
        half = max_pieces // 2
        out = out[:half] + out[-(max_pieces - half):]
    return out


def _action(block):
    """Short description of a tool call, or None."""
    name = block.get("name", "")
    inp = block.get("input") or {}
    if not isinstance(inp, dict):
        return None
    short = name.split("__")[-1]
    if inp.get("description") and isinstance(inp["description"], str):
        return inp["description"][:160]
    if inp.get("title") and isinstance(inp["title"], str):
        return f"{short}: {inp['title'][:160]}"
    if short in ("WebSearch",) and inp.get("query"):
        return f"search: {inp['query'][:160]}"
    if short == "Skill" and inp.get("skill"):
        return f"skill {inp['skill']} {str(inp.get('args', ''))[:120]}".strip()
    if short in ("Write", "Edit") and inp.get("file_path"):
        return f"{short} {Path(inp['file_path']).name}"
    if short == "Bash" and isinstance(inp.get("command"), str):
        m = GH_TITLE_RE.search(inp["command"])
        if m:
            return f"gh create: {m.group(2)[:160]}"
    return None


def parse_file(path):
    """Read one session transcript into session metadata + turns."""
    sid = Path(path).stem
    meta = {"session_id": sid, "path": str(path), "cwd": None, "title": None,
            "ai_title": None, "branch": None, "forked_from": None,
            "first_ts": None, "last_ts": None, "entrypoint": None}
    turns, links = [], {}
    cur = None
    cwd_counts = defaultdict(int)

    def new_turn(ts, prompt, kind="prompt"):
        nonlocal cur
        cur = {"ts": ts, "kind": kind, "prompt": prompt, "texts": [], "actions": []}
        turns.append(cur)

    with open(path, "rb") as fh:
        for line in fh:
            # cheap pre-filter: skip record kinds we never read
            head = line[:200]
            if b'"type":"attachment"' in head or b'"type":"file-history' in head:
                continue
            try:
                d = _loads(line)
            except Exception:
                continue
            t = d.get("type")
            if t == "custom-title":
                meta["title"] = d.get("customTitle") or meta["title"]
                continue
            if t == "ai-title":
                meta["ai_title"] = d.get("aiTitle") or d.get("title") or meta["ai_title"]
                continue
            if t not in ("user", "assistant") or d.get("isSidechain"):
                continue
            ts = d.get("timestamp")
            if ts:
                meta["first_ts"] = meta["first_ts"] or ts
                meta["last_ts"] = ts
            if d.get("cwd"):
                cwd_counts[d["cwd"]] += 1
            meta["branch"] = d.get("gitBranch") or meta["branch"]
            meta["forked_from"] = meta["forked_from"] or (d.get("forkedFrom") or {}).get("sessionId") \
                if isinstance(d.get("forkedFrom"), dict) else meta["forked_from"] or d.get("forkedFrom")
            meta["entrypoint"] = meta["entrypoint"] or d.get("entrypoint")
            msg = d.get("message") or {}
            content = msg.get("content")

            if t == "user":
                if d.get("isMeta") or d.get("toolUseResult") is not None:
                    continue
                if isinstance(content, list) and any(
                    isinstance(c, dict) and c.get("type") == "tool_result" for c in content
                ):
                    continue
                raw = _text_of(content)
                if not raw.strip():
                    continue
                if d.get("isCompactSummary") or raw.startswith(CONTINUED):
                    new_turn(ts, raw[:20000], kind="summary")
                    continue
                p = clean_prompt(raw)
                if not p or p.startswith("[Request interrupted"):
                    continue
                new_turn(ts, p[:4000])
            else:  # assistant
                if not isinstance(content, list) or cur is None:
                    continue
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    if c.get("type") == "text" and c.get("text"):
                        cur["texts"].append(c["text"])
                        for u in LINK_RE.findall(c["text"]):
                            links.setdefault(u, ts)
                    elif c.get("type") == "tool_use":
                        a = _action(c)
                        if a:
                            cur["actions"].append(a)

    if cwd_counts:
        meta["cwd"] = max(cwd_counts, key=cwd_counts.get)
    return meta, turns, links


def turn_chunks(turn):
    """(kind, text) chunks to index for one turn."""
    out = []
    if turn["kind"] == "summary":
        body = turn["prompt"].split("Summary:", 1)[-1]
        return [("summary", p) for p in pieces(body, max_pieces=12)]
    if len(turn["prompt"]) >= 12:
        out += [("prompt", p) for p in pieces(turn["prompt"], max_pieces=3)]
    if turn["texts"]:
        final = turn["texts"][-1]
        earlier = " ".join(turn["texts"][:-1])
        out += [("reply", p) for p in pieces(final, max_pieces=4)]
        if len(earlier) > 40:
            out.append(("reply", " ".join(earlier.split())[:PIECE]))
    if turn["actions"]:
        acts = "; ".join(dict.fromkeys(turn["actions"]))
        out += [("action", p) for p in pieces(acts, max_pieces=2)]
    return out


# --------------------------------------------------------------------------- index

_model = None


def model():
    global _model
    if _model is None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        from huggingface_hub import snapshot_download
        from model2vec import StaticModel

        # huggingface_hub reads HF_HUB_OFFLINE once at import, so toggling it can't express
        # "cache first, network on a miss"; resolve the folder explicitly and load from disk
        want = ["*.json", "*.safetensors", "*.txt"]  # not the onnx copy of the weights
        try:
            path = snapshot_download(MODEL_NAME, allow_patterns=want, local_files_only=True)
        except Exception:
            print(f"ccs: downloading the embedding model {MODEL_NAME} (first run only)…",
                  file=sys.stderr)
            path = snapshot_download(MODEL_NAME, allow_patterns=want)
        _model = StaticModel.from_pretrained(path)
    return _model


def embed(texts):
    v = model().encode(texts, show_progress_bar=False).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v.astype(np.float16)


def connect():
    CACHE.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
        CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, mtime REAL, size INTEGER);
        CREATE TABLE IF NOT EXISTS sessions(
            session_id TEXT PRIMARY KEY, path TEXT, cwd TEXT, title TEXT, ai_title TEXT,
            branch TEXT, forked_from TEXT, first_ts TEXT, last_ts TEXT, entrypoint TEXT,
            n_prompts INTEGER, first_prompt TEXT);
        CREATE TABLE IF NOT EXISTS turns(
            id INTEGER PRIMARY KEY, session_id TEXT, ts TEXT, kind TEXT, prompt TEXT, reply TEXT);
        CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY, turn_id INTEGER, session_id TEXT, kind TEXT, text TEXT, vec BLOB);
        CREATE TABLE IF NOT EXISTS links(session_id TEXT, url TEXT, ts TEXT);
        CREATE INDEX IF NOT EXISTS turns_s ON turns(session_id);
        CREATE INDEX IF NOT EXISTS chunks_s ON chunks(session_id);
        CREATE INDEX IF NOT EXISTS links_s ON links(session_id);
        CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(text, tokenize='porter unicode61');
        CREATE VIRTUAL TABLE IF NOT EXISTS words USING fts5(
            text, content='', contentless_delete=1, tokenize='unicode61');
        CREATE VIRTUAL TABLE IF NOT EXISTS words_vocab USING fts5vocab(words, 'row');
        """
    )
    if "archived" not in {r[1] for r in db.execute("PRAGMA table_info(sessions)")}:
        db.execute("ALTER TABLE sessions ADD COLUMN archived INTEGER DEFAULT 0")
    row = db.execute("SELECT v FROM kv WHERE k='schema'").fetchone()
    if row and row[0] != SCHEMA_VERSION:
        # archived sessions exist nowhere else on disk; never wipe them for a rebuild
        if db.execute("SELECT count(*) FROM sessions WHERE archived=1").fetchone()[0]:
            sys.exit(f"ccs: index schema changed but {DB_PATH} holds sessions whose transcripts "
                     "Claude Code already deleted; migrate it instead of rebuilding")
        db.close()
        DB_PATH.unlink()
        for ext in ("-wal", "-shm"):
            Path(str(DB_PATH) + ext).unlink(missing_ok=True)
        return connect()
    if not row:
        # commit now: an open write transaction would hold the lock for this process's lifetime
        db.execute("INSERT INTO kv VALUES('schema', ?)", (SCHEMA_VERSION,))
        db.commit()
    return db


def _drop_session(db, sid):
    ids = [r[0] for r in db.execute("SELECT id FROM chunks WHERE session_id=?", (sid,))]
    db.executemany("DELETE FROM fts WHERE rowid=?", [(i,) for i in ids])
    db.executemany("DELETE FROM words WHERE rowid=?", [(i,) for i in ids])
    for tbl in ("chunks", "turns", "links", "sessions"):
        db.execute(f"DELETE FROM {tbl} WHERE session_id=?", (sid,))


def update_index(db, verbose=True, force=False):
    """Re-read only transcripts whose mtime/size changed. Returns #files re-read."""
    current = {}
    for p in PROJECTS.glob("*/*.jsonl"):
        try:
            st = p.stat()
        except OSError:
            continue
        current[str(p)] = (st.st_mtime, st.st_size)
    known = {r[0]: (r[1], r[2]) for r in db.execute("SELECT path, mtime, size FROM files")}
    stale = [p for p, s in current.items() if force or known.get(p) != s]
    gone = [p for p in known if p not in current]
    for p in gone:
        # Claude Code deletes transcripts after cleanupPeriodDays (default 30); keep what we
        # indexed so the session stays searchable, it just can't be resumed any more
        db.execute("UPDATE sessions SET archived=1 WHERE path=?", (p,))
        db.execute("DELETE FROM files WHERE path=?", (p,))
    if not stale:
        db.commit()
        return 0
    if verbose and len(stale) > 5:
        print(f"ccs: indexing {len(stale)} changed transcript(s)…", file=sys.stderr)
    stale.sort(key=lambda p: -current[p][1])  # big files first for better pool balance
    workers = min(8, os.cpu_count() or 2, len(stale))
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(parse_file, stale, chunksize=1))
    else:
        results = [parse_file(p) for p in stale]

    for path, (meta, turns, links) in zip(stale, results):
        sid = meta["session_id"]
        _drop_session(db, sid)
        prompts = [t for t in turns if t["kind"] == "prompt"]
        if prompts:  # skip empty / tool-only transcripts
            db.execute(
                "INSERT INTO sessions(session_id, path, cwd, title, ai_title, branch, forked_from, "
                "first_ts, last_ts, entrypoint, n_prompts, first_prompt) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (sid, path, meta["cwd"], meta["title"], meta["ai_title"], meta["branch"],
                 meta["forked_from"], meta["first_ts"], meta["last_ts"], meta["entrypoint"],
                 len(prompts), prompts[0]["prompt"][:300]),
            )
            batch = []
            for t in turns:
                reply = t["texts"][-1][:3000] if t["texts"] else ""
                cur = db.execute(
                    "INSERT INTO turns(session_id, ts, kind, prompt, reply) VALUES(?,?,?,?,?)",
                    (sid, t["ts"], t["kind"], t["prompt"][:4000], reply),
                )
                tid = cur.lastrowid
                batch += [(tid, k, txt) for k, txt in turn_chunks(t)]
            title = meta["title"] or meta["ai_title"]
            if title:
                batch.append((None, "title", title))
            if batch:
                vecs = embed([b[2] for b in batch])
                for (tid, kind, txt), v in zip(batch, vecs):
                    cid = db.execute(
                        "INSERT INTO chunks(turn_id, session_id, kind, text, vec) VALUES(?,?,?,?,?)",
                        (tid, sid, kind, txt, v.tobytes()),
                    ).lastrowid
                    db.execute("INSERT INTO fts(rowid, text) VALUES(?,?)", (cid, txt))
                    db.execute("INSERT INTO words(rowid, text) VALUES(?,?)", (cid, txt))
            db.executemany("INSERT INTO links VALUES(?,?,?)", [(sid, u, ts) for u, ts in links.items()])
        db.execute("INSERT OR REPLACE INTO files VALUES(?,?,?)", (path, *current[path]))
    db.commit()
    return len(stale)


# --------------------------------------------------------------------------- search

def _df(db, term):
    row = db.execute("SELECT doc FROM words_vocab WHERE term=?", (term,)).fetchone()
    return row[0] if row else 0


def _inflection(a, b):
    """connector/connectors, read/reads: the stemmer already handles these."""
    return len(os.path.commonprefix([a, b])) >= min(len(a), len(b)) - 2


def understand(db, query):
    """Query words, plus spelling fixes for words that are rare in the corpus but close to a
    common one ("conectors" -> "connectors"). Returns (words, {typo: fix})."""
    words = re.findall(r"[a-z0-9][a-z0-9_]*", query.lower())
    fixes = {}
    for w in words:
        if w in STOP or len(w) < 4 or not w.isalpha():
            continue
        df_w = _df(db, w)
        cands = dict(db.execute(
            "SELECT term, doc FROM words_vocab WHERE term >= ? AND term < ? "
            "AND length(term) BETWEEN ? AND ? AND doc >= 3",
            (w[0], chr(ord(w[0]) + 1), len(w) - 2, len(w) + 2),
        ).fetchall())
        close = [t for t in difflib.get_close_matches(w, list(cands), n=4, cutoff=0.8)
                 if t != w and t not in STOP and not _inflection(w, t)]
        if not close:
            continue
        best = max(close, key=cands.get)
        ratio = difflib.SequenceMatcher(None, w, best).ratio()
        if df_w <= 1 or (ratio >= 0.88 and cands[best] >= 10 * df_w):
            fixes[w] = best
    return words, fixes


def allowed_sessions(db, project=None, since=None, until=None, exclude=None):
    """Session ids passing the filters, or None when there are no filters."""
    where, params = [], []
    if project:
        where.append("(s.cwd LIKE ? OR s.title LIKE ? OR s.ai_title LIKE ?)")
        params += [f"%{project}%"] * 3
    if since:
        where.append("s.last_ts >= ?")
        params.append(since)
    if until:
        where.append("s.first_ts <= ?")
        params.append(until + "T23:59:59")
    if exclude:
        where.append("s.session_id != ?")
        params.append(exclude)
    if not where:
        return None
    return {r[0] for r in db.execute(
        "SELECT session_id FROM sessions s WHERE " + " AND ".join(where), params)}


def search(db, query, k=8, project=None, since=None, until=None, exclude=None):
    words, fixes = understand(db, query)
    # never drop the user's word; a correction is only ever added
    content = [w for w in words if w not in STOP] + list(fixes.values())
    # keep the typo too: your own prompts are spelled the way you type
    terms = list(dict.fromkeys(t for w in words if w not in STOP and len(w) > 1
                               for t in (w, fixes.get(w, w))))
    dense_text = " ".join(content) or query
    allowed = allowed_sessions(db, project, since, until, exclude)

    rows = db.execute("SELECT id, session_id, turn_id, kind, vec FROM chunks").fetchall()
    if not rows:
        return [], fixes
    ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
    mat = np.frombuffer(b"".join(r[4] for r in rows), dtype=np.float16).reshape(len(rows), -1)
    info = {r[0]: (r[1], r[2], r[3]) for r in rows}

    # dense ranking
    q = embed([dense_text])[0].astype(np.float32)
    sims = mat.astype(np.float32) @ q
    order = np.argsort(-sims)[:800]
    dense_rank = {int(ids[i]): r for r, i in enumerate(order)}

    # lexical ranking (BM25, porter-stemmed)
    lex_rank = {}
    if terms:
        fts_q = " OR ".join('"' + t.replace('"', "") + '"' for t in terms)
        for r, (cid,) in enumerate(db.execute(
            "SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT 800", (fts_q,)
        )):
            lex_rank[cid] = r

    # reciprocal-rank fusion per chunk, then max per turn (a long reply split into
    # pieces must not count as several hits)
    kind_w = {"prompt": 1.0, "reply": 0.9, "summary": 0.85, "action": 0.8, "title": 1.0}
    turn_best = {}
    for cid in set(dense_rank) | set(lex_rank):
        sid, tid, kind = info[cid]
        if allowed is not None and sid not in allowed:
            continue
        s = 0.0
        if cid in dense_rank:
            s += 1.0 / (RRF_K + dense_rank[cid])
        if cid in lex_rank:
            s += 1.0 / (RRF_K + lex_rank[cid])
        s *= kind_w.get(kind, 0.8)
        key = (sid, tid if tid is not None else f"title:{sid}")
        if s > turn_best.get(key, (0, None))[0]:
            turn_best[key] = (s, tid)

    per_session = defaultdict(list)
    for (sid, _), (s, tid) in turn_best.items():
        per_session[sid].append((s, tid))
    ranked = []
    for sid, lst in per_session.items():
        lst.sort(key=lambda x: -x[0])
        # the best moment dominates; a couple more relevant moments add a little,
        # but a 700-prompt session must not win on volume alone
        agg = sum(s * w for (s, _), w in zip(lst, AGG_W))
        ranked.append((agg, sid, lst))
    ranked.sort(key=lambda x: -x[0])

    results = []
    for agg, sid, lst in ranked[:k]:
        moments = [tid for _, tid in lst if tid is not None][:4]
        results.append({"score": agg, "session_id": sid, "moments": [(0, t) for t in moments]})
    return results, fixes


# --------------------------------------------------------------------------- output

def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() and not os.environ.get("NO_COLOR") else s


def _local(ts, fmt="%b %d %H:%M"):
    if not ts:
        return "?"
    try:
        return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone().strftime(fmt)
    except ValueError:
        return ts[:16]


def _home(p):
    return (p or "?").replace(str(Path.home()), "~", 1)


def hydrate(db, results):
    for r in results:
        if "remote" in r:  # synced from another machine: only what HydraDB returned
            info, mom = r.pop("remote"), sorted(r.pop("remote_moments"), key=lambda m: m["ts"] or "")
            ts = [m["ts"] for m in mom if m["ts"]]
            r.update({"cwd": info["project"], "title": info["name"], "ai_title": None,
                      "first_ts": min(ts, default=None), "last_ts": max(ts, default=None),
                      "n_prompts": None, "first_prompt": None, "forked_from": None,
                      "branch": None, "archived": 0, "device": info["device"],
                      "name": info["name"] or "?", "moments": mom, "links": []})
            r["resume"] = hydra_mod().show_command(hydra_mod().remote_command(
                info["device"], info["project"], r["session_id"]))
            continue
        s = db.execute(
            "SELECT cwd, title, ai_title, first_ts, last_ts, n_prompts, first_prompt, forked_from, "
            "branch, archived FROM sessions WHERE session_id=?", (r["session_id"],)).fetchone()
        r.update(dict(zip(("cwd", "title", "ai_title", "first_ts", "last_ts", "n_prompts",
                           "first_prompt", "forked_from", "branch", "archived"), s)))
        r["name"] = r["title"] or r["ai_title"] or (r["first_prompt"] or "")[:60]
        mom = []
        for _, tid in r["moments"]:
            t = db.execute("SELECT ts, kind, prompt, reply FROM turns WHERE id=?", (tid,)).fetchone()
            if t:
                mom.append({"ts": t[0], "kind": t[1], "prompt": t[2], "reply": t[3]})
        mom.sort(key=lambda m: m["ts"] or "")
        r["moments"] = mom
        r["links"] = [u for (u,) in db.execute(
            "SELECT url FROM links WHERE session_id=? ORDER BY ts", (r["session_id"],))]
        r["resume"] = f"cd {r['cwd']} && claude --resume {r['session_id']}"
    return collapse_forks(results)


def collapse_forks(results):
    """A forked/branched session repeats its parent's history, so both match on the same
    moments. Keep the better-ranked one and list the other as a fork."""
    kept = []
    for r in results:
        mts = {(m["ts"], m["prompt"][:80]) for m in r["moments"]}
        twin = next((k for k in kept if mts and mts <= k["_mts"]), None)
        if twin:
            twin["forks"].append(r["session_id"])
            continue
        r["_mts"], r["forks"] = mts, []
        kept.append(r)
    for r in kept:
        del r["_mts"]
    return kept


def show(results, query_shown):
    if not results:
        print("no matches")
        return
    print(_c("2", f"query: {query_shown}"))
    for i, r in enumerate(results, 1):
        span = _local(r["first_ts"], "%b %d")
        last = _local(r["last_ts"], "%b %d")
        if last != span:
            span += f" → {last}"
        print()
        size = f"on {r['device']}" if r.get("device") else f"{r['n_prompts']} prompts"
        where = f"{span} · {size} · {_home(r['cwd'])}"
        print(f"{_c('1;36', f'{i:>2}.')} {_c('1', r['name'])}  {_c('2', where)}")
        if r.get("why"):
            print(f"    {_c('33', '★ ' + r['why'])}")
        for m in r["moments"][:3]:
            if m["kind"] == "summary":
                txt = "[compacted-session summary]"
            else:
                txt = m["prompt"]
            txt = txt if len(txt) <= 150 else txt[:147] + "…"
            print(f"    {_c('2', _local(m['ts']))}  {txt}")
        if r.get("device"):
            print(f"    {_c('32', r['resume'])}")
        elif r.get("archived"):
            print(f"    {_c('2', 'transcript deleted by Claude Code cleanup; searchable, not resumable')}")
        else:
            print(f"    {_c('32', 'claude --resume ' + r['session_id'])}")
        for f in r.get("forks", []):
            print(f"    {_c('2', 'also in fork: claude --resume ' + f)}")


def pick_and_resume(results):
    try:
        choice = input(_c("1", "\nresume # (Enter to quit): ")).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not choice.isdigit() or not 1 <= int(choice) <= len(results):
        return
    r = results[int(choice) - 1]
    if r.get("device"):
        cmd = hydra_mod().remote_command(r["device"], r["cwd"], r["session_id"])
        print(_c("2", f"→ {r['resume']}"))
        os.execvp(cmd[0], cmd)
    if r.get("archived"):
        print("that transcript was deleted by Claude Code's cleanup, so it can't be resumed")
        return
    cwd = r["cwd"]
    if not cwd or not os.path.isdir(cwd):
        print(f"original directory {cwd!r} is gone; resuming from here may not find the session")
    else:
        os.chdir(cwd)
    print(_c("2", f"→ cd {_home(cwd)} && claude --resume {r['session_id']}"))
    os.execvp("claude", ["claude", "--resume", r["session_id"]])


# --------------------------------------------------------------------------- Claude rerank

RERANK_SYS = (
    "You help a developer find one of their past Claude Code sessions. You get their "
    "description and numbered candidate sessions (title, project dir, dates, opening "
    "prompts, the moments that matched, links created). Judge which sessions actually "
    "match what they describe, not just shared keywords. Reply with ONLY a JSON array, best "
    'first, of the matching candidates: [{"n": <number>, "why": "<one short sentence citing '
    'the specific moment/date>"}]. Omit candidates that do not match. Empty array if none.'
)


def _digest(i, r):
    lines = [f"### {i}. {r['name']}",
             f"dir: {_home(r['cwd'])} | {_local(r['first_ts'], '%Y-%m-%d')} → "
             f"{_local(r['last_ts'], '%Y-%m-%d')} | "
             + (f"on {r['device']}" if r.get("device") else f"{r['n_prompts']} prompts")]
    if r["first_prompt"]:
        lines.append(f"first prompt: {r['first_prompt'][:200]}")
    for m in r["moments"]:
        p = "[summary] " + m["prompt"][:400] if m["kind"] == "summary" else m["prompt"][:300]
        lines.append(f"- {_local(m['ts'])} user: {p}")
        if m["reply"]:
            lines.append(f"  claude: {' '.join(m['reply'].split())[:350]}")
    slugs = [u.rsplit("/", 1)[-1] for u in r["links"] if "linear.app" in u or "/pull/" in u][:8]
    if slugs:
        lines.append("links: " + ", ".join(slugs))
    return "\n".join(lines)


def rerank(query, results, model_name):
    prompt = f"What I'm looking for: {query}\n\nCandidates:\n\n" + "\n\n".join(
        _digest(i, r) for i, r in enumerate(results, 1))
    cmd = ["claude", "-p", "--model", model_name, "--no-session-persistence", "--tools", "",
           "--strict-mcp-config", "--disable-slash-commands", "--system-prompt", RERANK_SYS]
    try:
        out = subprocess.run(cmd, input=prompt, text=True, capture_output=True, timeout=240,
                             cwd=CACHE)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"ccs: Claude rerank failed ({e}); showing hybrid ranking", file=sys.stderr)
        return results
    m = re.search(r"\[.*\]", out.stdout, re.S)
    try:
        picks = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        picks = None
    if picks is None:
        print(f"ccs: could not parse Claude rerank output; showing hybrid ranking\n"
              f"{(out.stdout or out.stderr)[:300]}", file=sys.stderr)
        return results
    reordered = []
    for p in picks:
        n = p.get("n") if isinstance(p, dict) else None
        if isinstance(n, int) and 1 <= n <= len(results) and results[n - 1] not in reordered:
            results[n - 1]["why"] = p.get("why", "")
            reordered.append(results[n - 1])
    return reordered


# --------------------------------------------------------------------------- sync plumbing

SETTINGS = Path.home() / ".claude" / "settings.json"
HOOK_ARGS = "--sync --background"


def hydra_mod():
    import ccs_hydra

    return ccs_hydra


def install_hook(remove=False, settings=None):
    """Add (or remove) a SessionEnd hook that syncs to HydraDB when a Claude Code session ends."""
    settings = settings or SETTINGS
    try:
        cfg = json.loads(settings.read_text()) if settings.exists() else {}
    except json.JSONDecodeError as e:
        sys.exit(f"ccs: {settings} is not valid JSON ({e}); fix it first")
    groups = cfg.setdefault("hooks", {}).setdefault("SessionEnd", [])
    ours = lambda h: h.get("type") == "command" and HOOK_ARGS in h.get("command", "")  # noqa: E731
    for g in groups:
        g["hooks"] = [h for h in g.get("hooks", []) if not ours(h)]
    groups[:] = [g for g in groups if g.get("hooks")]
    if not remove:
        exe = shutil.which("ccs") or str(Path(sys.argv[0]).resolve())
        groups.append({"hooks": [{"type": "command", "command": f"{shlex.quote(exe)} {HOOK_ARGS}",
                                  "timeout": 30}]})
    if not groups:
        del cfg["hooks"]["SessionEnd"]
    if not cfg["hooks"]:
        del cfg["hooks"]
    if settings.exists():
        shutil.copy2(settings, settings.with_name(settings.name + ".ccs-backup"))
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps(cfg, indent=2) + "\n")
    return groups[-1]["hooks"][0]["command"] if not remove else None


def detach(log):
    """Fork into the background (double fork, new session) with output appended to `log`;
    the caller returns at once, so a SessionEnd hook never holds Claude Code's exit."""
    if os.fork():
        os._exit(0)
    os.setsid()
    if os.fork():
        os._exit(0)
    log.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    print(f"--- {dt.datetime.now().isoformat(timespec='seconds')} ccs {HOOK_ARGS}", flush=True)


# --------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="ccs", description="Search your Claude Code sessions in natural language.")
    ap.add_argument("query", nargs="*", help="what you remember about the session")
    ap.add_argument("-n", type=int, default=6, help="results to show (default 6)")
    ap.add_argument("-a", "--ask", action="store_true",
                    help="have Claude judge the top candidates and explain each match (slower)")
    ap.add_argument("--model", default="haiku", help="model for --ask (default haiku)")
    ap.add_argument("-p", "--project", help="only sessions whose dir or title contains this")
    ap.add_argument("--since", help="only sessions active on/after YYYY-MM-DD")
    ap.add_argument("--until", help="only sessions started on/before YYYY-MM-DD")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--include-current", action="store_true",
                    help="when run inside Claude Code, don't hide the session you're in")
    ap.add_argument("--no-pick", action="store_true", help="don't offer to resume")
    ap.add_argument("--reindex", action="store_true", help="rebuild the index from scratch")
    ap.add_argument("--stats", action="store_true", help="show index stats")
    hy = ap.add_argument_group("HydraDB backend (config: ~/.config/ccsearch/hydradb.env)")
    hy.add_argument("-H", "--hydra", action="store_true",
                    help="search with HydraDB instead of the local index")
    hy.add_argument("-t", "--thinking", action="store_true",
                    help="with -H: thinking mode (LLM query understanding + graph, slower)")
    hy.add_argument("--device", metavar="NAME",
                    help="only sessions from this machine (implies -H; see CCS_DEVICE)")
    hy.add_argument("--sync", action="store_true",
                    help="push new/changed turns to HydraDB, then show indexing status")
    hy.add_argument("--background", action="store_true",
                    help=f"with --sync: detach at once and log to {CACHE / 'sync.log'}")
    hy.add_argument("--install-hook", action="store_true",
                    help="sync automatically whenever a Claude Code session ends")
    hy.add_argument("--uninstall-hook", action="store_true", help="remove that hook")
    args = ap.parse_args(argv)
    if args.install_hook or args.uninstall_hook:
        cmd = install_hook(remove=args.uninstall_hook)
        print(f"ccs: removed the SessionEnd hook from {SETTINGS}" if args.uninstall_hook
              else f"ccs: added a SessionEnd hook to {SETTINGS}: {cmd}")
        return
    if args.background:
        if not args.sync:
            ap.error("--background goes with --sync")
        detach(CACHE / "sync.log")
    args.hydra = args.hydra or bool(args.device)

    db = connect()
    update_index(db, force=args.reindex)
    here = None if args.include_current else os.environ.get("CLAUDE_CODE_SESSION_ID")
    if args.stats or args.sync:
        n_s, n_t, n_c = (db.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                         for t in ("sessions", "turns", "chunks"))
        n_a = db.execute("SELECT count(*) FROM sessions WHERE archived=1").fetchone()[0]
        print(f"{n_s} sessions ({n_a} with deleted transcripts), {n_t} turns, {n_c} chunks · "
              f"{DB_PATH} ({DB_PATH.stat().st_size / 1e6:.0f} MB)")
    if args.sync:
        hydra = hydra_mod()
        print(hydra.sync(db, str(Path.home())))
        print("HydraDB indexing (sample):", hydra.indexing_report(db))
    query = " ".join(args.query).strip()
    if not query:
        if not (args.reindex or args.stats or args.sync):
            ap.print_help()
        return

    pool = max(args.n, 15) if args.ask else args.n
    if args.hydra:
        hydra = hydra_mod()
        allowed = allowed_sessions(db, args.project, args.since, args.until, here)
        if args.device and args.device != hydra.device_name():
            allowed = set()  # every session in the local index is from this machine

        def keep_remote(sid, info, moments):
            """The same filters for sessions synced from other machines, judged on what
            HydraDB returned: dates come from the matched turns only."""
            if sid == here or (args.device and info["device"] != args.device):
                return False
            if args.project and not any(args.project.lower() in (v or "").lower()
                                        for v in (info["project"], info["name"])):
                return False
            ts = [m["ts"] for m in moments if m["ts"]]
            return not ts or not ((args.since and max(ts) < args.since)
                                  or (args.until and min(ts) > args.until + "T23:59:59"))

        results, _ = hydra.search(db, query, k=pool * 2, allowed=allowed, keep_remote=keep_remote,
                                  mode="thinking" if args.thinking else "fast")
        fixes = {}
    else:
        results, fixes = search(db, query, k=pool * 2, project=args.project, since=args.since,
                                until=args.until, exclude=here)
    results = hydrate(db, results)
    if args.ask and results:
        print(_c("2", f"asking {args.model} to judge {len(results)} candidates…"), file=sys.stderr)
        results = rerank(query, results, args.model)
    results = results[: args.n]
    shown = query
    if fixes:
        shown += "  (also searched: " + ", ".join(f"{a}→{b}" for a, b in fixes.items()) + ")"

    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return
    show(results, shown)
    if results and not args.no_pick and sys.stdin.isatty() and sys.stdout.isatty():
        pick_and_resume(results)


if __name__ == "__main__":
    main()
