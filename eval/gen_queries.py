#!/usr/bin/env python3
"""Generate a known-answer query set from the indexed sessions.

For every session that is synced to HydraDB (so all backends see the same corpus), sample
a few turns spread over the session and have Claude write the queries you would type weeks
later: a `gist` per session, and a `vague` and a `specific` query per sampled turn. A third
of the vague queries get one typo (`typo`). Gold is every session holding the same prompt
(forks and copies count), and for a gist every session holding half of its prompts.

Raw generations are cached in gen_cache/<session>.json; delete one to regenerate it.
Writes queries_gen.jsonl. Sessions listed in exclude.txt (one id per line), or in
$CCS_EVAL_EXCLUDE (comma-separated), are never sources or gold: put the sessions where you
built or discussed the eval there, since they quote the queries.
"""

import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ccsearch as c  # noqa: E402
import ccs_hydra as hydra  # noqa: E402

# sessions that talk about the eval itself must never be sources or gold
EXCLUDE = {s.strip() for s in os.environ.get("CCS_EVAL_EXCLUDE", "").split(",") if s.strip()}
if (HERE / "exclude.txt").exists():
    EXCLUDE |= {ln.split("#")[0].strip() for ln in (HERE / "exclude.txt").read_text().splitlines()} - {""}
CACHE = HERE / "gen_cache"
MODEL = "sonnet"

SYS = """You write search queries for evaluating a tool that finds past Claude Code chat sessions.
The user is a software engineer. Weeks later they type what they half-remember to find the
chat again.

Write each query the way they would type it: terse, lowercase is fine, paraphrased from
memory. HARD word limits: gist and vague at most 12 words, specific at most 14. One idea per
query; if the session covered several topics, the gist names the main one. Rules:
- never copy a distinctive phrase of 3 or more words from the text
- no file paths, function or variable names, ids, ticket numbers or URLs
- describe the problem or goal in your own words rather than quoting it
- each query must point at THIS session/moment, not at any coding chat in general
- specific adds exactly one concrete detail they would remember (a tool, product, library,
  error kind or number), not several

Style examples (other sessions): "the chat where i set up the gpu box for experiments",
"why did the nightly ingest job get slower", "pr check failing on lint versions",
"script that uploads a model to s3".

Output only JSON:
{"gist": "<the whole session, e.g. 'the chat where i ...'>",
 "moments": [{"n": 1, "vague": "<only the gist of that moment>",
              "specific": "<same moment plus one concrete detail>"}]}
with one entry per numbered moment."""


def cut(s, n):
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[:n] + " …"


def eligible(turns, synced, sid):
    out = []
    for idx, (tid, ts, kind, prompt, reply) in enumerate(turns):
        p = (prompt or "").strip()
        if (kind == "prompt" and hydra.item_id(sid, idx) in synced and 25 <= len(p) <= 2500
                and len(reply or "") >= 150):
            out.append((tid, ts, p, reply))
    return out


def pick(cands, sid):
    n = min(8, max(1, round(len(cands) ** 0.5 / 2)))
    rnd = random.Random(sid)
    bins = [cands[i * len(cands) // n:(i + 1) * len(cands) // n] for i in range(n)]
    return [rnd.choice(b) for b in bins if b]


def prompt_for(meta, turns, picked):
    sid, cwd, title, ai_title, first_ts = meta
    prompts = [t[3] for t in turns if t[2] == "prompt" and len((t[3] or "").strip()) >= 25]
    summary = next((t[3].split("Summary:", 1)[-1] for t in turns if t[2] == "summary"), "")
    step = max(1, len(prompts) // 10)
    out = [f"Session title: {title or ai_title or '?'}",
           f"Started: {(first_ts or '')[:10]}   Project dir: {cwd}",
           "", "Opening prompts:"] + [f"- {cut(p, 300)}" for p in prompts[:3]]
    out += ["", "Prompts sampled across the session:"] + [f"- {cut(p, 150)}" for p in prompts[3::step][:10]]
    if summary:
        out += ["", "Summary of an earlier part:", cut(summary, 1500)]
    out += ["", "Moments to write queries for:"]
    for n, (tid, ts, p, reply) in enumerate(picked, 1):
        out += [f"[{n}] ({(ts or '')[:10]})", f"User: {cut(p, 1200)}", f"Claude: {cut(reply, 1200)}", ""]
    return "\n".join(out)


def generate(sid, text):
    f = CACHE / f"{sid}.json"
    if f.exists():
        return json.loads(f.read_text())
    cmd = ["claude", "-p", "--model", MODEL, "--no-session-persistence", "--tools", "",
           "--strict-mcp-config", "--disable-slash-commands", "--system-prompt", SYS]
    for _ in range(3):
        r = subprocess.run(cmd, input=text, text=True, capture_output=True, timeout=300, cwd=CACHE)
        m = re.search(r"\{.*\}", r.stdout, re.S)
        try:
            out = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            out = None
        if out and out.get("gist") and out.get("moments"):
            f.write_text(json.dumps(out, indent=1))
            return out
        print(f"  {sid[:8]}: bad output, retrying: {(r.stdout or r.stderr)[:200]}", file=sys.stderr)
    return None


def typo(q, rnd):
    words = [w for w in re.findall(r"[a-z]{5,}", q) if w not in c.STOP]
    if not words:
        return None
    w = rnd.choice(words)
    i = rnd.randrange(1, len(w) - 1)
    op = rnd.choice(("swap", "drop", "double"))
    bad = (w[:i] + w[i + 1] + w[i] + w[i + 2:] if op == "swap" else
           w[:i] + w[i + 1:] if op == "drop" else w[:i] + w[i] + w[i:])
    return q.replace(w, bad, 1) if bad != w else None


def main():
    CACHE.mkdir(exist_ok=True)
    db = c.connect()
    synced = {r[0] for r in db.execute("SELECT item_id FROM hydra_sync")}
    sessions = db.execute("SELECT session_id, cwd, title, ai_title, first_ts FROM sessions "
                          "ORDER BY first_ts").fetchall()
    turns = {s[0]: db.execute("SELECT id, ts, kind, prompt, reply FROM turns WHERE session_id=? "
                              "ORDER BY id", (s[0],)).fetchall() for s in sessions}
    # prompt text → sessions/turns holding it (forks and pasted copies repeat prompts)
    holders = defaultdict(set)
    for sid, ts_ in turns.items():
        if sid in EXCLUDE:
            continue
        for tid, _, kind, p, _ in ts_:
            if kind == "prompt" and len((p or "").strip()) >= 25:
                holders[p.strip()].add((sid, tid))
    own = {sid: {p.strip() for _, _, k, p, _ in ts_ if k == "prompt" and len((p or "").strip()) >= 25}
           for sid, ts_ in turns.items()}

    jobs = []
    for meta in sessions:
        sid = meta[0]
        if sid in EXCLUDE:
            continue
        cands = eligible(turns[sid], synced, sid)
        if not cands:
            continue
        picked = pick(cands, sid)
        jobs.append((meta, picked, prompt_for(meta, turns[sid], picked)))
    print(f"{len(jobs)} sessions, {sum(len(j[1]) for j in jobs)} moments", file=sys.stderr)

    with ThreadPoolExecutor(6) as ex:
        outs = list(ex.map(lambda j: generate(j[0][0], j[2]), jobs))

    rows, rnd = [], random.Random(0)
    for (meta, picked, _), out in zip(jobs, outs):
        sid = meta[0]
        if not out:
            print(f"  {sid[:8]}: no queries", file=sys.stderr)
            continue
        mine = own[sid]
        gist_gold = [sid] + [t for t in own if t != sid and t not in EXCLUDE and mine
                             and len(mine & own[t]) >= 0.5 * len(mine)]
        rows.append({"qid": f"{sid[:8]}-g", "kind": "gist", "query": out["gist"].strip(),
                     "gold": gist_gold, "src_session": sid})
        for m in out["moments"]:
            n = m.get("n")
            if not isinstance(n, int) or not 1 <= n <= len(picked):
                continue
            tid, _, p, reply = picked[n - 1]
            hold = holders.get(p, {(sid, tid)})
            gold = sorted({s for s, _ in hold}, key=lambda s: s != sid)
            base = {"gold": gold, "gold_turns": sorted(t for _, t in hold), "src_session": sid,
                    "src_turn": tid}
            for kind in ("vague", "specific"):
                if m.get(kind):
                    rows.append({"qid": f"{sid[:8]}-t{tid}-{kind[0]}", "kind": kind,
                                 "query": m[kind].strip(), **base})
            if m.get("vague") and rnd.random() < 1 / 3:
                t = typo(m["vague"].strip().lower(), rnd)
                if t:
                    rows.append({"qid": f"{sid[:8]}-t{tid}-x", "kind": "typo", "query": t, **base})
    with open(HERE / "queries_gen.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    kinds = defaultdict(int)
    for r in rows:
        kinds[r["kind"]] += 1
    print(f"wrote {len(rows)} queries: {dict(kinds)}", file=sys.stderr)


if __name__ == "__main__":
    main()
