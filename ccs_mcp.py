"""`ccs mcp`: serve past Claude Code sessions to Claude over MCP (stdio, JSON-RPC 2.0).

Two tools: `search_sessions` finds sessions, `read_session` reads the turns of one. A small
hand-written server instead of the MCP SDK, whose API changed between major versions: this only
needs initialize, tools/list and tools/call. Everything the tools print goes to stderr, because
stdout carries the protocol; every result is passed through the credential masking first.

Register it once:  claude mcp add --scope user ccs -- ccs mcp
"""

import contextlib
import json
import os
import sys
import traceback

from ccs_redact import redact

VERSION = "0.3.0"
INSTRUCTIONS = (
    "Search and read the user's past Claude Code sessions, from this machine and (with HydraDB "
    "sync) their other machines. Use it when the user refers to earlier work ('like last time', "
    "'the chat where we...', 'what did we decide about...') or when an earlier session would "
    "help: search first, then read the relevant turns of the best session."
)

TOOLS = [
    {
        "name": "search_sessions",
        "description": (
            "Find past Claude Code sessions by describing what happened in them. Returns the "
            "best sessions with their matching moments (what the user asked, what Claude "
            "replied), dates, directory or machine, and the command that resumes each. "
            "backend: 'auto' (default) uses HydraDB thinking mode when it is set up, which is "
            "best for vague or paraphrased descriptions; 'local' searches only this machine's "
            "index but includes sessions not synced yet (e.g. from today)."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "what the session was about"},
                "limit": {"type": "integer", "default": 6, "minimum": 1, "maximum": 20},
                "project": {"type": "string", "description": "only sessions whose directory or title contains this"},
                "since": {"type": "string", "description": "only sessions active on/after YYYY-MM-DD"},
                "until": {"type": "string", "description": "only sessions started on/before YYYY-MM-DD"},
                "backend": {"type": "string", "enum": ["auto", "local", "fast", "thinking"], "default": "auto"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_session",
        "description": (
            "Read turns of one past session (each: what the user asked and Claude's final "
            "reply) by session id or id prefix from search_sessions. Give `query` to get the "
            "turns most relevant to it (use the session's own vocabulary: names, PR numbers, "
            "errors), or `start` to page through the session in order. Works for sessions in "
            "this machine's index, including ones whose transcript Claude Code already deleted."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "query": {"type": "string"},
                "start": {"type": "integer", "default": 0, "minimum": 0},
                "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 40},
            },
            "required": ["session_id"],
        },
    },
]


class Args:
    """The CLI flags that search and the HydraDB filters read."""

    def __init__(self, project=None, since=None, until=None):
        self.project, self.since, self.until, self.device = project, since, until, None


def _clip(text, n):
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n - 1] + "…"


def search_sessions(core, db, a):
    query = (a.get("query") or "").strip()
    if not query:
        raise ValueError("query is empty")
    limit = max(1, min(int(a.get("limit") or 6), 20))
    backend = a.get("backend") or "auto"
    if backend == "auto":
        backend = "thinking" if core.hydra_configured() else "local"
    elif backend != "local" and not core.hydra_configured():
        raise ValueError(f"backend {backend!r} needs HydraDB (~/.config/ccsearch/hydradb.env)")
    args = Args(a.get("project"), a.get("since"), a.get("until"))
    here = os.environ.get("CLAUDE_CODE_SESSION_ID")
    if backend == "local":
        results, _ = core.search(db, query, k=limit * 2, project=args.project, since=args.since,
                                 until=args.until, exclude=here)
    else:
        allowed, keep_remote = core.hydra_filters(db, args, here)
        results, _ = core.hydra_mod().search(db, query, k=limit * 2, allowed=allowed,
                                             keep_remote=keep_remote, mode=backend)
    results = core.hydrate(db, results)[:limit]
    if not results:
        return f"No past sessions match {query!r} ({backend} search)."
    out = [f"{len(results)} sessions for {query!r} ({backend} search), best first:"]
    for i, r in enumerate(results, 1):
        span = core._local(r["first_ts"], "%Y-%m-%d")
        if r["last_ts"] and r["last_ts"][:10] != (r["first_ts"] or "")[:10]:
            span += " → " + core._local(r["last_ts"], "%Y-%m-%d")
        where = f"on machine {r['device']}" if r.get("device") else f"{r['n_prompts']} prompts"
        out.append(f"\n{i}. {r['name']}  [session {r['session_id']}]\n"
                   f"   {span} · {where} · {core._home(r['cwd'])}")
        if r.get("archived"):
            out.append("   transcript deleted by Claude Code's cleanup"
                       + ("; `ccs --restore` brings it back" if r.get("backup") else ""))
        else:
            out.append(f"   resume: {r['resume']}")
        for m in r["moments"][:3]:
            out.append(f"   - {core._local(m['ts'], '%Y-%m-%d %H:%M')} user: {_clip(m['prompt'], 300)}")
            if m.get("reply"):
                out.append(f"     claude: {_clip(m['reply'], 300)}")
    return "\n".join(out)


def read_session(core, db, a):
    sid = (a.get("session_id") or "").strip()
    if len(sid) < 4:
        raise ValueError("give at least 4 characters of the session id")
    limit = max(1, min(int(a.get("limit") or 10), 40))
    got = core.session_turns(db, sid, a.get("query"), max(0, int(a.get("start") or 0)), limit)
    if got is None:
        n = db.execute("SELECT count(*) FROM sessions WHERE session_id LIKE ?", (sid + "%",)).fetchone()[0]
        if n > 1:
            raise ValueError(f"{n} sessions start with {sid!r}; give more of the id")
        return (f"Session {sid!r} is not in this machine's index. If search_sessions showed it "
                "'on machine X', it lives on that machine: its excerpts in the search result are "
                "all that is available here.")
    (full, cwd, title, ai_title, first_ts, last_ts, n_prompts, first_prompt, archived), turns = got
    total = db.execute("SELECT count(*) FROM turns WHERE session_id=?", (full,)).fetchone()[0]
    name = title or ai_title or _clip(first_prompt, 60)
    how = f"turns most relevant to {a['query']!r}" if a.get("query") else "turns in order"
    out = [f"{name}  [session {full}]",
           f"{core._local(first_ts, '%Y-%m-%d')} → {core._local(last_ts, '%Y-%m-%d')} · "
           f"{total} turns · {core._home(cwd)}",
           ("transcript deleted by Claude Code's cleanup" if archived
            else f"resume: cd {cwd} && claude --resume {full}"),
           f"Showing {len(turns)} {how}; numbers are positions in the session (0-based)."]
    for pos, ts, kind, prompt, reply in turns:
        out.append(f"\n#{pos} · {core._local(ts, '%Y-%m-%d %H:%M')}")
        if kind == "summary":
            out.append("[summary of the session so far] " + _clip(prompt.split("Summary:", 1)[-1], 3000))
            continue
        out.append(f"User: {_clip(prompt, 1500)}")
        if reply:
            out.append(f"Claude: {reply[:2500]}" + ("…" if len(reply) > 2500 else ""))
    return "\n".join(out)


HANDLERS = {"search_sessions": search_sessions, "read_session": read_session}


def serve(core, stdin=None, stdout=None):
    """Answer JSON-RPC requests, one per line, until stdin closes."""
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    db = None

    def send(msg):
        stdout.write(json.dumps(msg) + "\n")
        stdout.flush()

    for line in stdin:
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
            continue
        mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if mid is None:  # a notification (initialized, cancelled, ...): nothing to answer
            continue
        if method == "initialize":
            result = {"protocolVersion": params.get("protocolVersion") or "2025-06-18",
                      "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": "ccs", "version": VERSION},
                      "instructions": INSTRUCTIONS}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call" and params.get("name") in HANDLERS:
            try:
                with contextlib.redirect_stdout(sys.stderr):  # keep the protocol stream clean
                    if db is None:
                        db = core.connect()
                    core.update_index(db, verbose=False)
                    text = HANDLERS[params["name"]](core, db, params.get("arguments") or {})
                result = {"content": [{"type": "text", "text": redact(text)}], "isError": False}
            except Exception as e:  # report to the model instead of dropping the connection
                traceback.print_exc(file=sys.stderr)
                result = {"content": [{"type": "text", "text": f"ccs error: {e}"}], "isError": True}
        elif method == "tools/call":
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32602, "message": f"unknown tool {params.get('name')!r}"}})
            continue
        else:
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601, "message": f"method not found: {method}"}})
            continue
        send({"jsonrpc": "2.0", "id": mid, "result": result})
