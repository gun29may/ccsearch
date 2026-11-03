import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ccsearch as c  # noqa: E402

WIFI = "aaaaaaaa-0000-0000-0000-000000000001"
PAY = "bbbbbbbb-0000-0000-0000-000000000002"
POEM = "cccccccc-0000-0000-0000-000000000003"


def write_session(proj, sid, turns, title=None, cwd="/work/app"):
    recs = [{"type": "custom-title", "customTitle": title}] if title else []
    for i, (prompt, reply) in enumerate(turns):
        recs.append({"type": "user", "timestamp": f"2026-10-01T10:{2 * i:02d}:00Z", "cwd": cwd,
                     "message": {"role": "user", "content": prompt}})
        recs.append({"type": "assistant", "timestamp": f"2026-10-01T10:{2 * i + 1:02d}:00Z",
                     "cwd": cwd, "message": {"role": "assistant",
                                             "content": [{"type": "text", "text": reply}]}})
    path = proj / f"{sid}.jsonl"
    path.write_text("\n".join(map(json.dumps, recs)) + "\n")
    return path


@pytest.fixture
def proj(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    (projects / "-work-app").mkdir(parents=True)
    monkeypatch.setattr(c, "PROJECTS", projects)
    monkeypatch.setattr(c, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(c, "DB_PATH", tmp_path / "cache" / "index.sqlite")
    monkeypatch.setattr(c, "BACKUP", tmp_path / "backup")
    return projects / "-work-app"


def test_clean_prompt_strips_harness_wrappers():
    assert c.clean_prompt("<system-reminder>noise</system-reminder> fix the   flaky test") == \
        "fix the flaky test"
    assert c.clean_prompt("<command-name>/review</command-name><command-args>42</command-args>") == \
        "/review 42"
    assert c.clean_prompt("<bash-input>git status</bash-input>") == "! git status"


def test_pieces_keep_opening_and_conclusion():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 17 for i in range(10))
    out = c.pieces(text, limit=250, max_pieces=4)
    assert len(out) == 4 and all(len(p) <= 250 for p in out)
    assert out[0].startswith("Paragraph 0") and "Paragraph 9" in out[-1]


def test_parse_file(tmp_path):
    recs = [
        {"type": "custom-title", "customTitle": "payments retry"},
        {"type": "user", "timestamp": "2026-10-01T10:00:00Z", "cwd": "/w/pay",
         "message": {"content": "<system-reminder>x</system-reminder>why does the retry test flake"}},
        {"type": "assistant", "timestamp": "2026-10-01T10:01:00Z", "message": {"content": [
            {"type": "tool_use", "name": "Bash",
             "input": {"command": "pytest", "description": "Run the retry tests"}},
            {"type": "text", "text": "The clock isn't frozen. https://github.com/acme/pay/pull/12"}]}},
        {"type": "user", "timestamp": "2026-10-01T10:02:00Z", "isSidechain": True,
         "message": {"content": "a subagent's prompt"}},
        {"type": "user", "timestamp": "2026-10-01T10:03:00Z", "isCompactSummary": True,
         "message": {"content": "Summary: we froze the clock"}},
    ]
    path = tmp_path / f"{PAY}.jsonl"
    path.write_text("\n".join(map(json.dumps, recs)))
    meta, turns, links = c.parse_file(path)
    assert meta["title"] == "payments retry" and meta["cwd"] == "/w/pay"
    assert [t["kind"] for t in turns] == ["prompt", "summary"]
    assert turns[0]["prompt"] == "why does the retry test flake"
    assert turns[0]["actions"] == ["Run the retry tests"]
    assert list(links) == ["https://github.com/acme/pay/pull/12"]


def test_item_ids_round_trip():
    hy = pytest.importorskip("ccs_hydra")
    assert hy.item_id("abc-1", 7) == "abc-1__t0007"
    assert hy.parse_item_id("abc-1__t0007") == ("abc-1", 7)
    assert hy.parse_item_id("not-an-item") == (None, None)


def test_index_search_typos_and_archive(proj):
    """End to end on a tiny corpus (loads the Model2Vec model)."""
    write_session(proj, WIFI, [
        ("my wifi keeps dropping after the laptop sleeps", "Reload the iwlwifi driver, then "
         "power-cycle: a warm reboot does not reset the card."),
    ], title="wifi drops")
    write_session(proj, PAY, [
        ("the payments retry test is flaky", "It sleeps on a real clock; freeze time instead."),
        ("payments client still times out in CI", "Raise the payments timeout in the CI profile."),
        ("add a payments regression test", "Added a test that replays the failed payments call."),
    ])
    write_session(proj, POEM, [("write a haiku about databases", "Rows in quiet night")])
    db = c.connect()
    assert c.update_index(db, verbose=False) == 3
    assert c.update_index(db, verbose=False) == 0  # incremental: nothing changed

    res, _ = c.search(db, "wireless keeps disconnecting after sleep", k=3)
    assert res[0]["session_id"] == WIFI

    res, fixes = c.search(db, "flaky paymnets test", k=3)
    assert fixes == {"paymnets": "payments"}
    assert res[0]["session_id"] == PAY

    res, _ = c.search(db, "flaky payments test", k=3, exclude=PAY)
    assert PAY not in {r["session_id"] for r in res}

    # Claude Code deleted the transcript: still searchable, marked archived
    (proj / f"{WIFI}.jsonl").unlink()
    c.update_index(db, verbose=False)
    res, _ = c.search(db, "wifi dropping after sleep", k=3)
    top = c.hydrate(db, res)[0]
    assert top["session_id"] == WIFI and top["archived"] == 1


# --------------------------------------------------------------------------- multi-device

REMOTE = "dddddddd-0000-0000-0000-000000000004"


def remote_chunks(sid=REMOTE, device="dgx", project="~/lab/frozen labels"):
    body = (f"Device: {device}\nSession: gpu label sweep\nProject: {project}\n\n"
            "User: rerun the label sweep on the spark\n\nClaude: Started 4 runs.")
    title = body[:80]
    first = {"id": f"{sid}__t0003", "chunk_uuid": f"{sid}__t0003_chunk_0000", "source_title": title,
             "chunk_content": "## Message Details\n\nFrom: me\nSent: 2026-10-05T09:00:00Z\n\n"
                              f"## Message\n\n{body}"}
    later = {"id": f"{sid}__t0007", "chunk_uuid": f"{sid}__t0007_chunk_0001", "source_title": title,
             "chunk_content": f"[claude_code] {title}\n\nrun 3 diverged after the lr warmup"}
    return [first, later]


@pytest.fixture
def db_one_local(proj):
    """An index holding one local session (no embeddings needed for HydraDB ranking)."""
    db = c.connect()
    db.execute("INSERT INTO sessions(session_id, path, cwd, title, first_ts, last_ts, n_prompts, "
               "first_prompt) VALUES(?,?,?,?,?,?,?,?)",
               (WIFI, "x", "/work/app", "wifi drops", "2026-10-01T10:00:00Z",
                "2026-10-01T10:05:00Z", 1, "my wifi keeps dropping"))
    db.execute("INSERT INTO turns(session_id, ts, kind, prompt, reply) VALUES(?,?,?,?,?)",
               (WIFI, "2026-10-01T10:00:00Z", "prompt", "my wifi keeps dropping",
                "Reload the driver, then power-cycle the card."))
    return db


def test_parse_chunk_first_and_later():
    hy = pytest.importorskip("ccs_hydra")
    first, later = map(hy.parse_chunk, remote_chunks())
    assert (first["device"], first["session"], first["project"]) == \
        ("dgx", "gpu label sweep", "~/lab/frozen labels")
    assert first["ts"] == "2026-10-05T09:00:00Z" and first["full"]
    assert first["prompt"] == "rerun the label sweep on the spark"
    assert later["device"] == "dgx" and not later["full"] and later["ts"] is None
    assert later["prompt"] == "… run 3 diverged after the lr warmup"


def test_rank_mixes_local_and_remote_sessions(db_one_local):
    hy = pytest.importorskip("ccs_hydra")
    local = {"id": f"{WIFI}__t0000", "chunk_content": "Device: laptop\nSession: wifi drops"}
    out = {r["session_id"]: r for r in hy.rank(db_one_local, [local] + remote_chunks(), k=5)}
    assert set(out) == {WIFI, REMOTE}
    assert out[WIFI]["moments"] and "remote" not in out[WIFI]
    assert out[REMOTE]["remote"] == {"device": "dgx", "name": "gpu label sweep",
                                     "project": "~/lab/frozen labels"}
    assert len(out[REMOTE]["remote_moments"]) == 2

    keep = lambda sid, info, moms: info["device"] == "laptop"  # noqa: E731
    assert [r["session_id"] for r in hy.rank(db_one_local, [local] + remote_chunks(), k=5,
                                              keep_remote=keep)] == [WIFI]
    assert hy.rank(db_one_local, [local] + remote_chunks(), k=5, allowed=set())[0]["session_id"] \
        == REMOTE


def test_remote_result_renders_and_resumes_over_ssh(db_one_local, capsys):
    hy = pytest.importorskip("ccs_hydra")
    res = c.hydrate(db_one_local, hy.rank(db_one_local, remote_chunks(), k=5))
    r = res[0]
    assert r["device"] == "dgx" and r["name"] == "gpu label sweep"
    assert r["first_ts"] == "2026-10-05T09:00:00Z"
    assert hy.remote_command("dgx", "~/lab/frozen labels", REMOTE) == \
        ["ssh", "-t", "dgx", f"cd ~/'lab/frozen labels' && claude --resume {REMOTE}"]
    assert r["resume"] == f"ssh -t dgx \"cd ~/'lab/frozen labels' && claude --resume {REMOTE}\""
    c.show(res, "label sweep")
    out = capsys.readouterr().out
    assert "on dgx" in out and f"claude --resume {REMOTE}" in out and "ssh -t dgx" in out


def test_items_lead_with_the_device(db_one_local, monkeypatch):
    hy = pytest.importorskip("ccs_hydra")
    monkeypatch.setenv("CCS_DEVICE", "laptop")
    monkeypatch.setattr(hy, "ENV_FILE", Path("/nonexistent"))
    (item,) = hy.build_items(db_one_local, "/home/me")
    assert item["fields"]["body"].startswith("Device: laptop\nSession: wifi drops\n")
    assert item["additional_metadata"]["device"] == "laptop"


def test_install_hook_is_idempotent_and_reversible(tmp_path):
    settings = tmp_path / "settings.json"
    other = {"type": "command", "command": "echo bye"}
    settings.write_text(json.dumps({"model": "opus", "hooks": {"SessionEnd": [{"hooks": [other]}]}}))
    c.install_hook(settings=settings)
    c.install_hook(settings=settings)  # a second install replaces, never duplicates
    cfg = json.loads(settings.read_text())
    cmds = [h["command"] for g in cfg["hooks"]["SessionEnd"] for h in g["hooks"]]
    assert cfg["model"] == "opus" and cmds[0] == "echo bye"
    assert len(cmds) == 2 and cmds[1].endswith("--sync --background")
    c.install_hook(remove=True, settings=settings)
    cfg = json.loads(settings.read_text())
    assert cfg["hooks"]["SessionEnd"] == [{"hooks": [other]}]
    assert (tmp_path / "settings.json.ccs-backup").exists()


def test_only_one_sync_at_a_time(proj):
    hy = pytest.importorskip("ccs_hydra")
    db = c.connect()
    first = hy._lock(db)
    assert first is not None and hy._lock(db) is None  # a second sync skips
    first.close()
    assert hy._lock(db) is not None


def test_sync_keeps_refused_deletions_for_the_next_run(proj, monkeypatch):
    hy = pytest.importorskip("ccs_hydra")
    db = c.connect()
    db.execute("CREATE TABLE IF NOT EXISTS hydra_sync(item_id TEXT PRIMARY KEY, hash TEXT)")
    db.executemany("INSERT INTO hydra_sync VALUES(?, 'h')", [("a__t0000",), ("b__t0000",), ("c__t0000",)])

    class FakeHydra:
        def delete(self, ids):  # what /context DELETE returns: HTTP 200 with per-id outcomes
            return {"success": False, "results": [
                {"id": "a__t0000", "deleted": True},
                {"id": "b__t0000", "deleted": False, "error": "Source not found"},
                {"id": "c__t0000", "deleted": False, "error": "Source is still processing"}]}

    monkeypatch.setattr(hy, "Hydra", FakeHydra)
    monkeypatch.setattr(hy, "build_items", lambda db, home, *rest: [])
    out = hy.sync(db, "/home/me")
    assert (out["deleted"], out["deferred"]) == (2, 1)
    assert [r[0] for r in db.execute("SELECT item_id FROM hydra_sync")] == ["c__t0000"]


# --------------------------------------------------------------------------- redaction

FAKE_KEY = "sk_live_" + "Ab3" * 10


def test_redact_masks_credentials_and_spares_code():
    from ccs_redact import redact
    for secret, kind in [(FAKE_KEY, "stripe-style"), ("sk-or-v1-" + "a1" * 32, "sk-key"),
                         ("ghp_" + "Z" * 36, "github-token"), ("AKIA" + "ABCDEFGHIJKLMNOP", "aws-key-id"),
                         ("hf_" + "q" * 34, "hf-token")]:
        assert redact(f"key {secret} here") == f"key [redacted:{kind}] here"
    assert redact("postgres://admin:hunter2pass@db:5432/x") == \
        "postgres://admin:[redacted:url-password]@db:5432/x"
    assert redact("OPENAI_API_KEY = 'abcdefghijklmnopqrstuv'") == \
        "OPENAI_API_KEY = '[redacted:assigned-secret]'"
    for text in ("pip install scikit-learn", "tokenizer_path = /models/tok/tokenizer.json",
                 "commit 3f2a9c10e4b8d7c6a5b4c3d2e1f0a9b8c7d6e5f4", "max_tokens: 2048",
                 "sk_live_... placeholder", "https://github.com/gun29may/ccsearch"):
        assert redact(text) == text


def test_sync_items_are_redacted_and_excludable(db_one_local, monkeypatch):
    hy = pytest.importorskip("ccs_hydra")
    from collections import Counter
    monkeypatch.setattr(hy, "ENV_FILE", Path("/nonexistent"))
    db_one_local.execute("UPDATE turns SET reply = ? WHERE session_id = ?",
                         (f"Use {FAKE_KEY} for the prod database.", WIFI))
    masked = Counter()
    (item,) = hy.build_items(db_one_local, "/home/me", masked)
    assert FAKE_KEY not in json.dumps(item) and "[redacted:stripe-style]" in item["fields"]["body"]
    assert masked == {"stripe-style": 1}
    for pattern in ("/work/app", "/work/*", WIFI[:8]):
        monkeypatch.setenv("CCS_SYNC_EXCLUDE", f"~/elsewhere, {pattern}")
        skipped = set()
        assert hy.build_items(db_one_local, "/home/me", None, skipped) == [] and skipped == {WIFI}
    monkeypatch.setenv("CCS_SYNC_EXCLUDE", "/work/application")  # a prefix, not the dir
    assert len(hy.build_items(db_one_local, "/home/me")) == 1


# --------------------------------------------------------------------------- backups

def test_backup_appends_and_restores_a_deleted_transcript(proj, tmp_path):
    db = c.connect()
    path = write_session(proj, POEM, [("write a haiku about databases", "Rows in quiet night")])
    current = {str(path): (path.stat().st_mtime, path.stat().st_size)}
    assert c.backup_transcripts(db, current, verbose=False) == 1
    first_gz = db.execute("SELECT gz_size FROM backups").fetchone()[0]

    with path.open("a") as fh:  # the session goes on: only the new bytes are added
        fh.write(json.dumps({"type": "user", "timestamp": "2026-10-02T10:00:00Z", "cwd": "/work/app",
                             "message": {"role": "user", "content": "now a limerick"}}) + "\n")
    current = {str(path): (path.stat().st_mtime, path.stat().st_size)}
    assert c.backup_transcripts(db, current, verbose=False) == 1
    assert c.backup_transcripts(db, current, verbose=False) == 0  # nothing new
    gz = tmp_path / "backup" / "-work-app" / f"{POEM}.jsonl.gz"
    assert gz.stat().st_size > first_gz
    original = path.read_bytes()
    import gzip
    assert gzip.decompress(gz.read_bytes()) == original  # members concatenate to the file

    path.unlink()  # Claude Code's cleanup
    restored, sid = c.restore(db, POEM[:8])
    assert sid == POEM and restored == path and path.read_bytes() == original


def test_backup_recopies_a_rewritten_transcript(proj, tmp_path):
    db = c.connect()
    path = write_session(proj, POEM, [("write a haiku about databases", "Rows in quiet night")])
    c.backup_transcripts(db, {str(path): (0, path.stat().st_size)}, verbose=False)
    path.write_text(path.read_text().replace("haiku", "sonnet") + "\n")  # not an append
    c.backup_transcripts(db, {str(path): (0, path.stat().st_size)}, verbose=False)
    import gzip
    gz = tmp_path / "backup" / "-work-app" / f"{POEM}.jsonl.gz"
    assert gzip.decompress(gz.read_bytes()) == path.read_bytes()


# --------------------------------------------------------------------------- ccs ask, ccs mcp

def test_chunk_text_drops_headers():
    first, later = remote_chunks()
    assert c._chunk_text(first) == "User: rerun the label sweep on the spark\n\nClaude: Started 4 runs."
    assert c._chunk_text(later) == "run 3 diverged after the lr warmup"


def test_cited_sources_group_by_session():
    ex = [{"session_id": s, "name": s[:4], "resume": "r", "archived": 0} for s in (WIFI, PAY, WIFI)]
    assert [(x["session_id"], x["n"]) for x in c.cited_sources("see [3] and [1], not [9]", ex)] == \
        [(WIFI, [1, 3])]
    assert [x["session_id"] for x in c.cited_sources("no citations", ex)] == [WIFI, PAY]


def test_ask_excerpts_are_local_turns_with_secrets_masked(proj):
    from types import SimpleNamespace
    write_session(proj, PAY, [("the payments retry test is flaky",
                               f"Freeze the clock; the CI key is {FAKE_KEY}.")])
    db = c.connect()
    c.update_index(db, verbose=False)
    args = SimpleNamespace(project=None, since=None, until=None, device=None)
    (e,) = c.gather(db, "flaky payments test", args, None, "local")
    assert e["session_id"] == PAY and e["text"].startswith("User: the payments retry test is flaky")
    assert FAKE_KEY not in e["text"] and "[redacted:stripe-style]" in e["text"]


def test_mcp_server_round_trip(proj):
    import io
    import ccs_mcp
    write_session(proj, WIFI, [("my wifi keeps dropping after the laptop sleeps",
                                f"Reload the driver. Token: {FAKE_KEY}")], title="wifi drops")
    write_session(proj, POEM, [("write a haiku about databases", "Rows in quiet night")])
    reqs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
                "name": "search_sessions", "arguments": {"query": "wifi dropping", "backend": "local"}}},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
                "name": "read_session", "arguments": {"session_id": WIFI[:8]}}},
            {"jsonrpc": "2.0", "id": 5, "method": "nope"}]
    out = io.StringIO()
    ccs_mcp.serve(c, io.StringIO("\n".join(map(json.dumps, reqs)) + "\n"), out)
    res = {r["id"]: r for r in map(json.loads, out.getvalue().splitlines())}
    assert sorted(res) == [1, 2, 3, 4, 5]  # the notification got no answer
    assert res[1]["result"]["protocolVersion"] == "2025-06-18"
    assert {t["name"] for t in res[2]["result"]["tools"]} == {"search_sessions", "read_session"}
    found = res[3]["result"]["content"][0]["text"]
    assert found.index(WIFI) < found.index(POEM) if POEM in found else WIFI in found
    turns = res[4]["result"]["content"][0]["text"]
    assert "my wifi keeps dropping" in turns and "[redacted:stripe-style]" in turns
    assert FAKE_KEY not in out.getvalue()
    assert res[5]["error"]["code"] == -32601
