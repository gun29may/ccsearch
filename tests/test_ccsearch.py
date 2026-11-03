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
    monkeypatch.setattr(hy, "build_items", lambda db, home: [])
    out = hy.sync(db, "/home/me")
    assert (out["deleted"], out["deferred"]) == (2, 1)
    assert [r[0] for r in db.execute("SELECT item_id FROM hydra_sync")] == ["c__t0000"]
