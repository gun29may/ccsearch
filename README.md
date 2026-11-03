# ccsearch

Find a past [Claude Code](https://claude.com/claude-code) session by describing it, then jump
back into it.

```console
$ ccs "the chat where the payments retry test kept flaking"
query: the chat where the payments retry test kept flaking

 1. payments retry  Oct 01 → Oct 03 · 14 prompts · ~/code/payments
    Oct 01 10:00  why does the retry test flake on CI but not locally
    Oct 03 16:42  freeze the clock in the retry tests and add a regression test
    claude --resume 3f2a9c10-6b1e-4c55-9d0a-2f8e1c7b9a41

 2. CI flakiness sweep  Sep 22 · 31 prompts · ~/code/payments
    Sep 22 11:05  list every test that failed more than twice this week
    claude --resume 77d0e4b2-1a9f-4e0b-8c3d-5b6a7f8e9d10

resume # (Enter to quit): 1
→ cd ~/code/payments && claude --resume 3f2a9c10-6b1e-4c55-9d0a-2f8e1c7b9a41
```

Claude Code keeps every session as a JSONL transcript under `~/.claude/projects/`, but
`claude --resume` only lists recent titles in the current directory. `ccs` indexes all of them and
searches what was said: your prompts, Claude's replies, the actions it took and the summaries
written when a session was compacted. It runs locally, needs no API key, and answers in about
0.2 s.

## Install

Needs Python 3.10+ whose SQLite is 3.43 or newer (check with
`python3 -c "import sqlite3; print(sqlite3.sqlite_version)"`), and the `claude` CLI for resuming
and for `--ask`.

```bash
pipx install "ccsearch[hydradb] @ git+https://github.com/gun29may/ccsearch"
```

or from a clone:

```bash
git clone https://github.com/gun29may/ccsearch && cd ccsearch
pip install -e ".[hydradb]"
```

Drop `[hydradb]` if you only want local search. The first run downloads the
[Model2Vec](https://github.com/MinishLab/model2vec) embedding model
`minishlab/potion-retrieval-32M` (about 125 MB) and indexes every transcript; after that
only changed transcripts are re-read.

## Usage

```bash
ccs "where did I decide to drop the dual-vector embedding idea"
ccs -a "the chat where I found all the connector issues"   # Claude judges the candidates
ccs -p payments --since 2026-10-01 "flaky retry test"      # filter by project and date
ccs --json "s3 upload script" | jq '.[0].resume'
```

| Flag | What it does |
|---|---|
| `-n N` | results to show (default 6) |
| `-a`, `--ask` | have Claude (`--model`, default `haiku`) read the top 15 candidates, drop the ones that only share keywords, and say why each match fits |
| `-p`, `--project TEXT` | only sessions whose directory or title contains `TEXT` |
| `--since` / `--until YYYY-MM-DD` | only sessions active on/after, or started on/before, a date |
| `--json` | machine-readable output, including the `cd … && claude --resume …` command |
| `--no-pick` | print results without offering to resume |
| `--include-current` | inside Claude Code the session you are in is hidden; this shows it |
| `--stats`, `--reindex` | index size, or a rebuild from scratch |
| `-H`, `-t`, `--sync` | search with HydraDB instead (see [below](#hydradb-backend-optional)) |

Picking a result runs `claude --resume <id>` in the session's original directory. Inside Claude
Code you can run it as `! ccs "…"`, or just ask Claude to find a past chat; it can call
`ccs --no-pick --json` itself.

### Keep your transcripts

Claude Code deletes transcripts 30 days after their last change (`cleanupPeriodDays`). `ccs` keeps
every session it has already indexed: when a transcript disappears the session is marked
archived, stays searchable, and shows as not resumable. A session that was deleted before `ccs`
ever indexed it is gone. To keep transcripts on disk longer, raise the limit in
`~/.claude/settings.json`:

```json
{ "cleanupPeriodDays": 3650 }
```

## How it works

**Indexing.** Each transcript becomes a list of turns: your prompt, with harness wrappers such as
system reminders stripped and pasted content shortened, Claude's final reply, one-line
descriptions of the tool calls it made, and any compaction summary. Turns are split into chunks
of up to 700 characters. Each chunk is embedded with Model2Vec and added to an SQLite FTS5 index
(Porter stemming). Subagent transcripts are skipped. Session titles and the GitHub PR, Linear
issue and Claude artifact links created in a session are kept for display. Everything lives in
`~/.cache/ccsearch/index.sqlite`.

**Search.**
1. **Typos.** Query words that are rare in your sessions but close to a common word are
   searched both ways ("conectors" also searches "connectors"). Corrections are added, never
   swapped in, because your own prompts are spelled the way you type.
2. **Ranking chunks.** BM25 and embedding similarity each rank chunks, and the two rankings are
   fused with reciprocal rank fusion (k = 30).
3. **Rolling up to turns and sessions.** A turn scores by its best chunk. A session scores by
   its best three turns, weighted 1, 0.35 and 0.15, so a 700-prompt session can't win on
   volume alone.
4. **Forks.** A session forked from another one repeats its parent's history, so the copy is
   folded into the better-ranked result.

**`--ask`** sends the top candidates (titles, dates, matching moments and links) to
`claude -p` with tools, MCP servers and session persistence turned off. The model returns the
matches in order with a one-line reason for each. It uses your Claude Code login.

## HydraDB backend (optional)

`ccs` can also store your sessions in [HydraDB](https://docs.hydradb.com) and search them with
its hosted retrieval. In thinking mode that means LLM query understanding, graph context and
reranking.

```bash
mkdir -p ~/.config/ccsearch && chmod 700 ~/.config/ccsearch
cat > ~/.config/ccsearch/hydradb.env <<'EOF'
HYDRADB_API_KEY=sk_live_...
HYDRADB_DATABASE=claude_sessions
HYDRADB_COLLECTION=yourname
EOF
chmod 600 ~/.config/ccsearch/hydradb.env

ccs --sync                       # upload new/changed turns (creates the database on first run)
ccs -H "flaky retry test"        # fast mode
ccs -H -t "flaky retry test"     # thinking mode
```

Environment variables of the same names override the file; `HYDRADB_BASE_URL` defaults to
`https://api.hydradb.com`. `--sync` uploads each turn as one `message` item with the id
`<session>__tNNNN`. Only new or changed turns are sent. Items of archived sessions are kept.

**What leaves your machine:** for every turn, the session title, project directory, your prompt,
Claude's final reply and the tool-action descriptions. That includes anything you pasted into
a prompt, such as keys or customer data. Search results are filtered and shown locally, just
like local search.

## Evaluation

`eval/` holds a known-answer benchmark you can build from your own sessions:

- `queries_hand.jsonl`: queries you write yourself, each with the session(s) it should find
  (see `queries_hand.example.jsonl`).
- `gen_queries.py`: has Claude (Sonnet) read turns sampled across every session and write
  queries the way you would type them weeks later: a gist per session, and a vague, a specific
  and a misspelled query per sampled turn. A query counts as answered by any session holding the
  same prompt, so forks count.
- `run_eval.py local|fast|thinking` scores each backend: MRR, recall@1/3, whether the exact
  source turn is among the moments shown, and latency. `--redo` reruns failed queries;
  `--report` prints a side-by-side table.

Put the sessions where you built or discussed the eval in `eval/exclude.txt`, since they quote
the queries. The generated data and results stay local (`.gitignore`).

Results on the author's sessions (32 sessions and about 3,300 turns, 2026-10-29):

| Queries | n | Local (MRR / R@1 / R@3) | HydraDB fast | HydraDB thinking |
|---|---|---|---|---|
| All | 262 | 0.874 / 0.80 / 0.94 | 0.897 / 0.84 / 0.96 | **0.935 / 0.89 / 0.98** |
| Hand-written | 15 | 0.756 / 0.60 / 0.87 | 0.832 / 0.73 / 0.93 | **0.889 / 0.80 / 1.00** |
| Median latency | | 0.16 s | 1.3 s | 3.7 s |

- **Thinking is the clear winner.** It beats local (+0.061 MRR, 95% CI +0.030 to +0.094) and
  fast (+0.038).
- **Fast vs local is too close to call** (+0.024, CI −0.009 to +0.057).
- **The generated queries are easier than real ones.** Despite instructions to paraphrase,
  they reuse about three quarters of their words from the source turn. The hand-written row is
  closer to real use.

## Configuration

| Variable | Default | |
|---|---|---|
| `CCS_PROJECTS` | `~/.claude/projects` | where transcripts are read from |
| `CCS_CACHE` | `~/.cache/ccsearch` | where the index lives |
| `HYDRADB_*` | `~/.config/ccsearch/hydradb.env` | HydraDB backend, see above |

## Development

```bash
pip install -e ".[hydradb,dev]"
ruff check . && pytest -q
```

The end-to-end test indexes three synthetic sessions and checks ranking, typo correction,
session exclusion and archiving; it loads the embedding model.

## License

MIT
