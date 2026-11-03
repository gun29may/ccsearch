"""HydraDB backend for ccs: push session turns as app knowledge, search them with /query.

Each turn of the local index becomes one `message` item whose id is
`<session_id>__t<NNNN>` (turn order within the transcript, stable across re-parses),
so a returned chunk maps straight back to a session and a turn without a lookup.
Config comes from ~/.config/ccsearch/hydradb.env (HYDRADB_API_KEY, HYDRADB_BASE_URL,
HYDRADB_DATABASE, HYDRADB_COLLECTION, CCS_DEVICE); environment variables override the file.

Several machines can sync into one collection. Every item starts with a `Device:` line, since
/query returns neither our title nor our metadata, only the text, and later chunks of a long
turn carry just its first 80 characters. A hit from a session that is not in the local index
is rebuilt from that text, and it resumes over ssh on the machine it came from.
"""

import getpass
import hashlib
import json
import os
import random
import re
import shlex
import socket
import sys
import time
from pathlib import Path

import httpx

ENV_FILE = Path.home() / ".config" / "ccsearch" / "hydradb.env"
BATCH = 50
BODY_MAX = 12000


def settings():
    cfg = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    for k in ("HYDRADB_API_KEY", "HYDRADB_BASE_URL", "HYDRADB_DATABASE", "HYDRADB_COLLECTION",
              "CCS_DEVICE"):
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    return cfg


def device_name():
    """This machine's name in the shared collection; also the ssh host other machines use."""
    return settings().get("CCS_DEVICE") or socket.gethostname().split(".")[0]


def config():
    cfg = settings()
    if not cfg.get("HYDRADB_API_KEY"):
        sys.exit(f"ccs: no HYDRADB_API_KEY (set it in {ENV_FILE})")
    cfg.setdefault("HYDRADB_BASE_URL", "https://api.hydradb.com")
    cfg.setdefault("HYDRADB_DATABASE", "claude_sessions")
    cfg.setdefault("HYDRADB_COLLECTION", getpass.getuser())
    return cfg


class Hydra:
    def __init__(self):
        c = config()
        self.db, self.coll = c["HYDRADB_DATABASE"], c["HYDRADB_COLLECTION"]
        self.http = httpx.Client(base_url=c["HYDRADB_BASE_URL"].rstrip("/"), timeout=120,
                                 headers={"Authorization": f"Bearer {c['HYDRADB_API_KEY']}"})

    def _post(self, url, **kw):
        """Retry 429 with linear backoff and transport errors with exponential backoff."""
        transport = 0
        for attempt in range(1, 11):
            try:
                r = self.http.post(url, **kw)
            except httpx.TransportError as e:
                transport += 1
                if transport > 6:
                    raise
                time.sleep(min(120, 5 * 2 ** (transport - 1)))
                print(f"  transport error ({type(e).__name__}), retrying", file=sys.stderr)
                continue
            if r.status_code != 429:
                return r
            time.sleep(min(60, 5 * attempt))
        raise RuntimeError(f"gave up on 429s for POST {url}")

    def ensure_database(self):
        """Create the database as `split` (the default `unified` layout rejects app knowledge)."""
        r = self._post("/databases", json={"database": self.db, "type": "split"})
        if r.status_code >= 400 and "already exist" not in r.text.lower():
            raise RuntimeError(f"create database {r.status_code}: {r.text[:300]}")

    def ingest(self, items):
        r = self._post("/context/ingest", data={
            "type": "knowledge", "database": self.db, "collection": self.coll,
            "upsert": "true", "infer": "true", "app_knowledge": json.dumps(items)})
        if r.status_code >= 400:
            raise RuntimeError(f"ingest {r.status_code}: {r.text[:400]}")
        return r.json().get("data") or {}

    def delete(self, ids):
        r = self.http.request("DELETE", "/context", json={
            "database": self.db, "collection": self.coll, "type": "knowledge", "ids": ids})
        r.raise_for_status()
        return r.json().get("data") or {}

    def status(self, ids):
        out = []
        for i in range(0, len(ids), 50):
            r = self.http.get("/context/status", params={
                "database": self.db, "collection": self.coll, "ids": ids[i:i + 50]})
            r.raise_for_status()
            out += (r.json().get("data") or {}).get("statuses") or []
        return out

    def query(self, text, mode="fast", max_results=100, recency_bias=None):
        body = {"database": self.db, "collection": self.coll, "query": text,
                "type": "knowledge", "mode": mode, "max_results": max_results,
                "response_format": "structured"}
        if recency_bias is not None:
            body["recency_bias"] = recency_bias
        r = self._post("/query", json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"query {r.status_code}: {r.text[:300]}")
        j = r.json()
        return (j.get("data") or {}).get("chunks") or [], (j.get("meta") or {})


# --------------------------------------------------------------------------- items

def item_id(sid, idx):
    return f"{sid}__t{idx:04d}"


def parse_item_id(iid):
    sid, _, t = iid.rpartition("__t")
    return (sid, int(t)) if sid and t.isdigit() else (None, None)


def build_items(db, home):
    """One app-knowledge item per meaningful turn of every indexed session."""
    sessions = db.execute(
        "SELECT session_id, cwd, title, ai_title, first_prompt FROM sessions").fetchall()
    actions = {}
    for tid, text in db.execute("SELECT turn_id, text FROM chunks WHERE kind='action'"):
        actions.setdefault(tid, []).append(text)
    items, device = [], device_name()
    for sid, cwd, title, ai_title, first_prompt in sessions:
        name = title or ai_title or (first_prompt or "")[:60]
        project = (cwd or "?").replace(home, "~", 1)
        turns = db.execute("SELECT id, ts, kind, prompt, reply FROM turns "
                           "WHERE session_id=? ORDER BY id", (sid,)).fetchall()
        for idx, (tid, ts, kind, prompt, reply) in enumerate(turns):
            if kind == "summary":
                body = ("Summary of the earlier part of this session:\n"
                        + prompt.split("Summary:", 1)[-1].strip())
            else:
                if len(prompt) + len(reply or "") < 40:
                    continue
                body = f"User: {prompt}"
                if reply:
                    body += f"\n\nClaude: {reply}"
                if actions.get(tid):
                    body += "\n\nActions: " + "; ".join(actions[tid])
            body = f"Device: {device}\nSession: {name}\nProject: {project}\n\n{body}"[:BODY_MAX]
            iid = item_id(sid, idx)
            items.append({
                "id": iid, "title": f"{name} — {(ts or '')[:10]}",
                "kind": "message", "provider": "claude_code", "external_id": iid,
                "timestamp": ts,
                "fields": {"kind": "message", "body": body, "author": getpass.getuser(),
                           "thread_id": sid, "created_at": ts},
                "additional_metadata": {"session_id": sid, "project": project[:200],
                                        "device": device},
            })
    return items


def _hash(item):
    return hashlib.sha1(json.dumps(item, sort_keys=True).encode()).hexdigest()


def _lock(db):
    """Exclusive, non-blocking lock next to the index, or None if another sync holds it."""
    path = Path(db.execute("PRAGMA database_list").fetchone()[2] + ".sync-lock")
    fh = path.open("w")
    try:
        import fcntl

        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except ImportError:  # no fcntl (Windows): run unlocked
        pass
    except OSError:
        fh.close()
        return None
    return fh


def sync(db, home, dry_run=False, exclude=None):
    """Push new/changed turns, delete items whose turn disappeared. Returns counts."""
    lock = _lock(db)
    if lock is None:
        print("ccs sync: another sync is running; skipping", file=sys.stderr)
        return {"sent": 0, "deleted": 0, "failed": 0, "skipped": "locked"}
    try:
        return _sync(db, home, dry_run, exclude)
    finally:
        lock.close()


def _sync(db, home, dry_run, exclude):
    db.execute("CREATE TABLE IF NOT EXISTS hydra_sync(item_id TEXT PRIMARY KEY, hash TEXT)")
    sent = dict(db.execute("SELECT item_id, hash FROM hydra_sync"))
    items = [it for it in build_items(db, home)
             if not exclude or not it["id"].startswith(exclude)]
    want = {it["id"]: _hash(it) for it in items}
    todo = [it for it in items if sent.get(it["id"]) != want[it["id"]]]
    gone = [iid for iid in sent if iid not in want]
    print(f"ccs sync ({device_name()}) → {len(items)} turns, {len(todo)} new/changed, "
          f"{len(gone)} to delete", file=sys.stderr)
    if dry_run or (not todo and not gone):
        return {"sent": 0, "deleted": 0, "failed": 0}
    h = Hydra()
    if not sent:
        h.ensure_database()
    failed = 0
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        res = h.ingest(batch)
        ok = {r["id"] for r in res.get("results") or [] if r.get("status") == "queued"}
        bad = [r for r in res.get("results") or [] if r.get("status") != "queued"]
        failed += len(bad)
        for r in bad[:3]:
            print(f"  failed {r.get('id')}: {r.get('error_code')} {r.get('error')}", file=sys.stderr)
        db.executemany("INSERT OR REPLACE INTO hydra_sync VALUES(?,?)",
                       [(it["id"], want[it["id"]]) for it in batch if it["id"] in ok])
        db.commit()
        print(f"  {min(i + BATCH, len(todo))}/{len(todo)} queued", file=sys.stderr)
        time.sleep(1)
    deleted, deferred = 0, []
    for i in range(0, len(gone), 100):
        # a 200 can still refuse some ids ("Source is still processing"); forget only the ones
        # that are really gone, so the rest are retried by the next sync instead of orphaned
        res = h.delete(gone[i:i + 100])
        done = [r["id"] for r in res.get("results") or []
                if r.get("deleted") or "not found" in (r.get("error") or "").lower()]
        deferred += [r for r in res.get("results") or [] if r["id"] not in set(done)]
        db.executemany("DELETE FROM hydra_sync WHERE item_id=?", [(g,) for g in done])
        db.commit()
        deleted += len(done)
    if deferred:
        print(f"  {len(deferred)} deletion(s) left for the next sync: {deferred[0].get('error')}",
              file=sys.stderr)
    return {"sent": len(todo) - failed, "deleted": deleted, "deferred": len(deferred),
            "failed": failed}


def indexing_report(db, sample=200):
    """Indexing status of a random sample of synced items."""
    ids = [r[0] for r in db.execute("SELECT item_id FROM hydra_sync")]
    if not ids:
        return {}
    counts = {}
    for s in Hydra().status(random.sample(ids, min(sample, len(ids)))):
        st = s.get("indexing_status") or "?"
        counts[st] = counts.get(st, 0) + 1
    return counts


# --------------------------------------------------------------------------- search

def search(db, query, k, mode="fast", allowed=None, keep_remote=None, rrf_k=30,
           agg_w=(1.0, 0.35, 0.15), recency_bias=None):
    """Same result shape as ccsearch.search: sessions with their best-matching turns."""
    chunks, meta = Hydra().query(query, mode=mode, recency_bias=recency_bias)
    return rank(db, chunks, k, allowed, keep_remote, rrf_k, agg_w), meta


_FIELD_RE = {f: re.compile(rf"^(?:\[claude_code\] )?{f.title()}: (.+)$", re.M)
             for f in ("device", "session", "project")}
_SENT_RE = re.compile(r"^Sent: (\S+)$", re.M)
_PROMPT_RE = re.compile(r"^User: (.*?)(?:\n\nClaude: |\n\nActions: |\Z)", re.M | re.S)


def parse_chunk(chunk):
    """Rebuild a turn from a returned chunk's text: the first chunk of an item holds the whole
    head ("## Message Details", "Sent:", then our body); later ones hold `[claude_code] ` +
    the body's first 80 characters (= source_title) + their own slice of the body."""
    text = chunk.get("chunk_content") or ""
    out = {f: (m.group(1).strip() if (m := r.search(text)) else None)
           for f, r in _FIELD_RE.items()}
    sent = _SENT_RE.search(text)
    out["ts"] = sent.group(1) if sent else None
    prefix = "[claude_code] " + (chunk.get("source_title") or "")
    if text.startswith("[claude_code] "):  # a later chunk: the prompt itself isn't in it
        out["kind"], out["full"] = "prompt", False
        out["prompt"] = "… " + " ".join(text[len(prefix):].split())[:300]
    elif "Summary of the earlier part of this session:" in text:
        out["kind"], out["full"] = "summary", True
        out["prompt"] = "[compacted-session summary]"
    else:
        m = _PROMPT_RE.search(text)
        out["kind"], out["full"] = "prompt", True
        out["prompt"] = " ".join(m.group(1).split())[:300] if m else ""
    return out


def remote_command(device, project, sid):
    """ssh command that resumes a session in its directory on the machine it lives on."""
    if project and project.startswith("~"):
        rest = project[1:].lstrip("/")
        where = "~/" + shlex.quote(rest) if rest else "~"  # the remote shell expands ~
    else:
        where = shlex.quote(project or ".")
    return ["ssh", "-t", device, f"cd {where} && claude --resume {sid}"]


def show_command(cmd):
    """`ssh -t host "cd ... && claude --resume id"`: same meaning as shlex.join, easier to read."""
    *head, last = cmd
    if not any(ch in last for ch in '"$`\\!'):
        return shlex.join(head) + f' "{last}"'
    return shlex.join(cmd)


def rank(db, chunks, k, allowed=None, keep_remote=None, rrf_k=30, agg_w=(1.0, 0.35, 0.15)):
    """Fold ranked chunks into sessions, scored like the local index.

    Sessions in the local index come back as turn ids; `allowed` (a set, or None for all)
    filters them. Sessions synced from other machines come back with `remote` (device, name,
    project) and `remote_moments` rebuilt from the chunk text; `keep_remote(sid, info,
    moments)` filters them.
    """
    local = {r[0] for r in db.execute("SELECT session_id FROM sessions")}
    seen, per_session, info, moments = set(), {}, {}, {}
    pos = 0
    for c in chunks:
        sid, idx = parse_item_id(c.get("id") or "")
        if sid is None or (sid, idx) in seen:
            continue
        seen.add((sid, idx))
        pos += 1
        if sid in local:
            if allowed is not None and sid not in allowed:
                continue
        else:
            p = parse_chunk(c)
            meta = info.setdefault(sid, {"device": None, "name": None, "project": None})
            for f, key in (("device", "device"), ("session", "name"), ("project", "project")):
                # the first chunk has the full value, later ones may cut it at 80 characters
                if p[f] and (p["full"] or not meta[key]) and len(p[f]) >= len(meta[key] or ""):
                    meta[key] = p[f]
            moments[(sid, idx)] = p
        per_session.setdefault(sid, []).append((1.0 / (rrf_k + pos), idx))
    results = []
    for sid, lst in per_session.items():
        if sid not in local:
            moms = [moments[(sid, i)] for _, i in lst]
            if not info[sid]["device"] or (keep_remote and not keep_remote(sid, info[sid], moms)):
                continue
        agg = sum(s * w for (s, _), w in zip(lst, agg_w))
        results.append((agg, sid, lst))
    results.sort(key=lambda x: -x[0])
    out, turn_ids = [], {}
    for agg, sid, lst in results[:k]:
        if sid not in local:
            moms = [{"ts": m["ts"], "kind": m["kind"], "prompt": m["prompt"], "reply": ""}
                    for m in (moments[(sid, i)] for _, i in lst[:4])]
            out.append({"score": agg, "session_id": sid, "moments": [],
                        "remote": info[sid], "remote_moments": moms})
            continue
        if sid not in turn_ids:
            turn_ids[sid] = [r[0] for r in db.execute(
                "SELECT id FROM turns WHERE session_id=? ORDER BY id", (sid,))]
        tids = [turn_ids[sid][i] for _, i in lst if i < len(turn_ids[sid])][:4]
        out.append({"score": agg, "session_id": sid, "moments": [(0, t) for t in tids]})
    return out
