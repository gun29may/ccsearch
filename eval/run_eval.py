#!/usr/bin/env python3
"""Run the known-answer query set against one backend, or report saved runs side by side.

    run_eval.py local|fast|thinking [--kinds hand,gist] [--limit N]
    run_eval.py --redo results/<run>.jsonl  rerun that run's errored/empty queries in place
    run_eval.py --report [result files]    (default: the newest run of each backend)

Both backends search the same corpus: sessions synced to HydraDB, minus the sessions that
discuss the eval itself. A query scores the rank of its first gold session; for queries
generated from a turn, `moment` says whether that turn is among the gold session's shown moments.
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ccsearch as c  # noqa: E402
import ccs_hydra as hydra  # noqa: E402
from gen_queries import EXCLUDE  # noqa: E402

RESULTS = HERE / "results"
K = 15
KIND_ORDER = ["hand", "gist", "vague", "specific", "typo"]


def load(kinds=None):
    rows = []
    for name in ("queries_hand.jsonl", "queries_gen.jsonl"):
        p = HERE / name
        if p.exists():
            rows += [json.loads(line) for line in p.open()]
    return [r for r in rows if not kinds or r["kind"] in kinds]


def overlap(query, text):
    """Share of the query's content words that appear in the source turn (leakage gauge)."""
    words = [w for w in re.findall(r"[a-z0-9]{3,}", query.lower()) if w not in c.STOP]
    have = set(re.findall(r"[a-z0-9]{3,}", text.lower()))
    return sum(w in have for w in words) / len(words) if words else 0.0


def score(q, results, have):
    gold = q["gold"]
    hit = lambda sid: any(sid.startswith(g) for g in gold)  # noqa: E731
    rank = next((i for i, r in enumerate(results, 1) if hit(r["session_id"])), None)
    moment = None
    if q.get("gold_turns") and rank:
        shown = {t for _, t in results[rank - 1]["moments"]}
        moment = bool(shown & set(q["gold_turns"]))
    return {"qid": q["qid"], "kind": q["kind"], "query": q["query"], "rank": rank,
            "moment": moment, "answerable": any(any(s.startswith(g) for s in have) for g in gold),
            "top5": [r["session_id"][:8] for r in results[:5]]}


def run(backend, queries, workers=4, path=None):
    db = c.connect()
    excl = EXCLUDE | {os.environ.get("CLAUDE_CODE_SESSION_ID", "")}
    synced = {r[0].rpartition("__t")[0] for r in db.execute("SELECT item_id FROM hydra_sync")}
    allowed = synced - excl
    have = {r[0] for r in db.execute("SELECT session_id FROM sessions")} & allowed
    src = {}
    for q in queries:
        if q.get("src_turn"):
            p, rep = db.execute("SELECT prompt, reply FROM turns WHERE id=?", (q["src_turn"],)).fetchone()
            src[q["qid"]] = f"{p}\n{rep}"
    out = []
    if backend == "local":
        c.model()
        for i, q in enumerate(queries, 1):
            t0 = time.time()
            res, _ = c.search(db, q["query"], k=60)
            res = [r for r in res if r["session_id"] in allowed][:K]
            out.append({**score(q, res, have), "latency": round(time.time() - t0, 2)})
            if i % 25 == 0:
                print(f"  {i}/{len(queries)}", file=sys.stderr)
    else:
        def ask(q):
            t0 = time.time()
            for attempt in range(3):
                try:
                    chunks, meta = hydra.Hydra().query(q["query"], mode=backend)
                    return chunks, meta.get("request_id"), round(time.time() - t0, 2), None
                except Exception as e:  # noqa: BLE001 — record and retry, report at the end
                    err = f"{type(e).__name__}: {e}"[:200]
                    time.sleep(5 * (attempt + 1))
            return [], None, round(time.time() - t0, 2), err
        with ThreadPoolExecutor(workers) as ex:
            for i, (q, (chunks, rid, lat, err)) in enumerate(zip(queries, ex.map(ask, queries)), 1):
                res = hydra.rank(db, chunks, K, allowed=allowed)
                out.append({**score(q, res, have), "latency": lat, "error": err,
                            "empty": not chunks, "request_id": rid})
                if i % 25 == 0:
                    print(f"  {i}/{len(queries)}", file=sys.stderr)
    for o in out:
        if o["qid"] in src:
            o["overlap"] = round(overlap(o["query"], src[o["qid"]]), 2)
    RESULTS.mkdir(exist_ok=True)
    if path:  # --redo: replace the failed rows, keep the rest
        redone = {o["qid"]: o for o in out}
        out = [redone.get(r["qid"], r) for r in map(json.loads, path.open())]
    path = path or RESULTS / f"{time.strftime('%Y%m%d-%H%M%S')}-{backend}.jsonl"
    with path.open("w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    bad = [o for o in out if o.get("error") or o.get("empty")]
    print(f"{backend}: {len(out)} queries → {path.name}; errors/empty: {len(bad)}", file=sys.stderr)
    return path


def metrics(rows):
    rows = [r for r in rows if r["answerable"]]
    n = len(rows)
    if not n:
        return None
    rr = [1 / r["rank"] if r["rank"] else 0 for r in rows]
    mom = [r["moment"] for r in rows if r["moment"] is not None and r["rank"] and r["rank"] <= 3]
    return {"n": n, "mrr": sum(rr) / n, "r1": sum(x == 1 for x in rr) / n,
            "r3": sum(x >= 1 / 3 for x in rr) / n, "r10": sum(x >= 0.1 for x in rr) / n,
            "moment": sum(mom) / len(mom) if mom else None,
            "lat": sorted(r["latency"] for r in rows)[n // 2]}


def report(paths):
    runs = {}
    for p in paths:
        backend = p.stem.split("-", 2)[-1]
        runs[backend] = ([json.loads(line) for line in p.open()], p.name)
    for b, (rows, name) in runs.items():
        bad = sum(bool(r.get("error") or r.get("empty")) for r in rows)
        print(f"{b:9} {name}  n={len(rows)} errors/empty={bad}")
    slices = [("all", lambda r: True), ("generated", lambda r: r["kind"] != "hand")]
    slices += [(k, lambda r, k=k: r["kind"] == k) for k in KIND_ORDER]
    slices += [("overlap<.5", lambda r: r.get("overlap") is not None and r["overlap"] < 0.5)]
    print(f"\n{'slice':11} {'n':>4}  " + "  ".join(f"{b:^30}" for b in runs))
    print(f"{'':11} {'':>4}  " + "  ".join(f"{'MRR   R@1  R@3  R@10 moment':30}" for _ in runs))
    for name, f in slices:
        cells, n = [], None
        for b, (rows, _) in runs.items():
            m = metrics([r for r in rows if f(r)])
            if not m:
                cells.append(f"{'-':^30}")
                continue
            n = m["n"]
            mom = f"{m['moment']:.2f}" if m["moment"] is not None else "  - "
            cells.append(f"{m['mrr']:.3f} {m['r1']:.2f} {m['r3']:.2f} {m['r10']:.2f}  {mom} ")
        if n:
            print(f"{name:11} {n:>4}  " + "  ".join(cells))
    print("\nmedian latency: " + ", ".join(
        f"{b} {metrics(rows)['lat']}s" for b, (rows, _) in runs.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", nargs="?", choices=["local", "fast", "thinking"])
    ap.add_argument("--kinds", help="comma list of " + ",".join(KIND_ORDER))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--redo", type=Path, help="result file whose failed queries to rerun")
    ap.add_argument("--report", nargs="*", help="result files (default: newest per backend)")
    a = ap.parse_args()
    if a.report is not None:
        paths = [Path(p) for p in a.report]
        if not paths:
            newest = {}
            for p in sorted(RESULTS.glob("*.jsonl")):
                newest[p.stem.split("-", 2)[-1]] = p
            paths = [newest[b] for b in ("local", "fast", "thinking") if b in newest]
        report(paths)
        return
    if a.redo:
        failed = {r["qid"] for r in map(json.loads, a.redo.open()) if r.get("error") or r.get("empty")}
        print(f"rerunning {len(failed)} failed queries", file=sys.stderr)
        run(a.redo.stem.split("-", 2)[-1], [q for q in load() if q["qid"] in failed], path=a.redo)
        return
    if not a.backend:
        ap.error("give a backend, --redo or --report")
    qs = load(a.kinds.split(",") if a.kinds else None)[:a.limit]
    run(a.backend, qs)


if __name__ == "__main__":
    main()
