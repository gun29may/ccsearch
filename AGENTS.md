# AGENTS.md

Notes for coding agents working on ccsearch: `ccs`, a CLI that searches Claude Code session
transcripts, answers questions from them, serves them over MCP, backs them up, and optionally
syncs them to HydraDB for hosted and cross-machine search. Read `README.md` for what users see;
this file covers how to change the code without breaking it.

## Commands

```bash
pip install -e ".[hydradb,dev]"   # Python 3.10+ whose SQLite is 3.43+ (FTS5 contentless_delete)
ruff check .                      # rule set pinned in pyproject.toml; CI installs the latest ruff
pytest -q                         # ~1.5 s once the embedding model is cached
```

- **What CI runs:** `ruff check .` and `pytest -q` on Python 3.10 and 3.12
  (`.github/workflows/ci.yml`). Code must parse on 3.10, so no PEP 701 f-strings (the same quote
  nested inside an f-string). `python3.10 -m py_compile *.py eval/*.py` catches that.
- **The end-to-end test** loads the Model2Vec model `minishlab/potion-retrieval-32M`, about 125
  MB, downloaded once.

## Layout

| File | What it holds |
|---|---|
| `ccsearch.py` | CLI (`main`), transcript parsing, the SQLite index, local hybrid search, display and resume, backups and `--restore`, `ccs ask`, hook install, `--background` |
| `ccs_hydra.py` | HydraDB backend: config, item building, `sync`, `/query`, `rank` (local and remote sessions), ssh resume command |
| `ccs_redact.py` | credential masking, used for everything that leaves the machine |
| `ccs_mcp.py` | `ccs mcp`: hand-written stdio JSON-RPC server (`search_sessions`, `read_session`) |
| `eval/` | known-answer eval: `gen_queries.py` (Claude writes queries), `run_eval.py local\|fast\|thinking`, `--report` |
| `tests/test_ccsearch.py` | all tests |

How the modules depend on each other:
- `ccsearch` imports `ccs_hydra` lazily through `hydra_mod()`, so local search works without
  httpx.
- `ccs_hydra` must not import `ccsearch`, because the CLI can run as `__main__` and the import
  would load a second copy. `ccs_mcp.serve()` takes the running `ccsearch` module as an argument
  for the same reason.

## Invariants: things that break silently

### Index

- **Location and schema.** The index is `~/.cache/ccsearch/index.sqlite` (`CCS_CACHE`), with
  its schema in `connect()`.
  - Bumping `SCHEMA_VERSION` deletes and rebuilds the index, and refuses when archived sessions
    exist, because their transcripts are gone.
  - Prefer additive migrations, like the `archived` column's `ALTER TABLE`.
- **`connect()` must not leave a write transaction open.** It once did, and a long-running
  process held the database lock for its whole life, so concurrent `ccs` processes failed with
  "database is locked".
- **`update_index` is incremental**, keyed on each transcript's (mtime, size).
  - A transcript that disappears marks its session `archived`; the session is never dropped.
    Claude Code deletes transcripts after `cleanupPeriodDays`, so this index may be the only
    copy left.

### Backups

- **Never delete backups.** They live in `~/.local/share/ccsearch/transcripts/<project dir>/<id>.jsonl.gz`.
- **They are append-only gzip members.** Before appending, the code checks the hash of the last
  4 KB backed up and the recorded `.gz` size. A mismatch means a full re-copy; don't weaken
  that check.

### HydraDB items

- **Item id is `<session_id>__tNNNN`**, where NNNN is the turn's position in its session,
  counting every turn, summaries included, in `turns.id` order.
  - `build_items`, `rank`, `ccs ask`'s `gather` and `eval/` all rely on this.
- **The item body starts with three header lines:**
  `Device: <name>\nSession: <title>\nProject: <~/dir>\n\n`, followed by
  `User: …`, `Claude: …` and `Actions: …`.
  - `/query` returns only text: `additional_metadata` comes back null, and `source_title` is
    just `body[:80]`.
  - The first chunk of an item is `## Message Details … ## Message\n\n<body>`. Later chunks are
    `[claude_code] ` + `body[:80]` + `\n\n` + their own slice.
  - `parse_chunk` and `_chunk_text` rebuild sessions from other machines out of that text, so
    keep `Device:` first and short.
  - Any change to the body changes every item's hash, so the next sync re-uploads every turn.
    Say so in the commit message.
- **Sync only records what really happened.**
  - Only ids whose ingest result came back `queued` count as sent.
  - `DELETE /context` answers HTTP 200 with a result per id. Forget an id only when it was
    deleted or is "not found". "Source is still processing" means retry on the next sync.
- **One sync at a time.** A sync takes `_lock()`, a non-blocking flock beside the index.

### Credentials and output

- **Mask everything that leaves the machine**: sync bodies and titles, `ccs ask` excerpts, MCP
  tool output.
  - Keep the patterns in `ccs_redact.py` narrow.
  - When you add a pattern, add a true-positive and a false-positive case to the tests.
- **Output streams:**
  - Results go to stdout; progress and errors go to stderr.
  - In `ccs mcp`, stdout is the protocol. Tool code runs under `redirect_stdout(sys.stderr)`,
    and nothing else may write to stdout.
- **Hide the session the user is in.** Exclude `CLAUDE_CODE_SESSION_ID` from results unless
  `--include-current` is set.

## Tests

- **Isolate every test from real data.** A test that indexes must use the `proj` fixture, which
  points `PROJECTS`, `CACHE`, `DB_PATH` and `BACKUP` at `tmp_path`. A test without it once wrote
  fake transcripts into the real backup folder.
- **Never commit real session content:** no excerpts, session ids, titles, home paths, ticket or
  PR numbers, in tests, the README or examples.
  - Build fake credentials by concatenation (`"sk_live_" + "Ab3" * 10`) so secret scanners
    don't flag them.
  - `eval/queries_*.jsonl`, `eval/gen_cache/`, `eval/results/` and `eval/exclude.txt` are
    generated from the user's own sessions and gitignored.
- **HydraDB code that can be unit-tested is tested against canned chunks** (`remote_chunks()`)
  and a fake `Hydra` class; see the existing tests.
- **To check against the real service:**
  - Create synthetic transcripts in sandboxes (`CCS_PROJECTS`, `CCS_CACHE` and `CCS_DEVICE` per
    fake machine) and sync them into a throwaway `HYDRADB_COLLECTION`.
  - Wait for indexing through `/context/status`, search, then delete the items. A delete can be
    refused while anything in the collection is still indexing; retry until it succeeds.
- **Never print the HydraDB key.** It lives in `~/.config/ccsearch/hydradb.env` (mode 600); read
  it through `ccs_hydra.config()`.

## Ranking changes

Measure before and after with the eval:

```bash
cd eval && python3 run_eval.py local && python3 run_eval.py fast && python3 run_eval.py thinking
python3 run_eval.py --report     # --redo results/<file> reruns only failed queries
```

- **Trust the hand-written slice most.** Claude-generated queries reuse about 75% of their words
  from the source turn, so they flatter lexical matching.
- **Thinking mode varies from run to run**; fast mode and local search don't.
- **Don't report a run with errors or empty results.**
- **The 2026-10-29 baseline** (MRR on all queries / on the hand-written ones):

  | Backend | All | Hand-written |
  |---|---|---|
  | local | 0.874 | 0.76 |
  | fast | 0.897 | 0.83 |
  | thinking | 0.935 | 0.89 |

## Style

- **Match the surrounding code:** small functions, docstrings that say why, and comments only
  where the reason isn't obvious.
- **Dependencies:** stdlib, numpy and model2vec, plus huggingface_hub; httpx is optional, for
  HydraDB. Don't add a dependency for something a few lines can do (the MCP server is
  hand-written for this reason: the SDK's API changed between major versions).
- **Model loading:** `model()` resolves the model folder with `snapshot_download` (cache first)
  and loads from disk. Toggling `HF_HUB_OFFLINE` after import does nothing, because
  huggingface_hub reads it once.
- **Python 3.10:** keep it working there, and test new flags through `main([...])`.
