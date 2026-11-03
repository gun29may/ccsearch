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
