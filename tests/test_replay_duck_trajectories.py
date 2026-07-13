from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import scripts.replay_duck_trajectories as replay_script
from drlqn.train.duck_replay import ReplayAlignmentError


class _DummyClient:
    def __init__(self, *_args, **_kwargs):
        pass

    def stats(self):
        return {"recordings": 2}

    def recordings(self, game_code=None):
        assert game_code is None
        return [
            {"id": "ok", "game_code": "aa00", "transition_count": 2},
            {"id": "bad", "game_code": "aa00", "transition_count": 2},
        ]

    def events(self, recording_id):
        return [{"recording_id": recording_id}]


class _DummyReport:
    def __init__(self, recording_id: str):
        self.recording_id = recording_id
        self.mismatches = ()

    def as_dict(self):
        return {"recording_id": self.recording_id, "mismatches": []}


def test_main_succeeds_when_some_recordings_replay(monkeypatch, tmp_path):
    report_payloads = []

    monkeypatch.setattr(
        replay_script,
        "_parse_args",
        lambda: SimpleNamespace(
            base_url="https://example.test/agi-3",
            api_key="test-key",
            game=[],
            recording=[],
            root=tmp_path / "root",
            seed=[],
            max_seed=2,
            allow_mismatch=False,
            output=tmp_path / "duck.npz",
            report=tmp_path / "duck_report.json",
        ),
    )
    monkeypatch.setattr(replay_script, "ArcDatasetClient", _DummyClient)
    monkeypatch.setattr(replay_script, "load_arcengine", lambda: object())
    monkeypatch.setattr(
        replay_script,
        "environment_files",
        lambda _root, _game_code: [Path("fake_env.py")],
    )
    monkeypatch.setattr(
        replay_script,
        "parse_recording_events",
        lambda _events, metadata: SimpleNamespace(
            recording_id=metadata["id"],
            game_code=metadata["game_code"],
        ),
    )

    def _replay(recording, **_kwargs):
        if recording.recording_id == "bad":
            raise ReplayAlignmentError("bad trace")
        return [object()], _DummyReport(recording.recording_id)

    monkeypatch.setattr(replay_script, "replay_recording", _replay)
    monkeypatch.setattr(
        replay_script,
        "save_examples_npz",
        lambda _examples, _output: _output,
    )
    monkeypatch.setattr(
        replay_script,
        "_write_report",
        lambda _path, payload: report_payloads.append(payload),
    )

    exit_code = replay_script.main()

    assert exit_code == 0
    assert report_payloads
    assert report_payloads[-1]["recordings_replayed"] == 1
    assert report_payloads[-1]["recordings_failed"] == 1
