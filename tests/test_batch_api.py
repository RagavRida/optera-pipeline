from __future__ import annotations

import json

from optera import batch_api


def test_create_batch_uses_independent_chat_completion_endpoint(monkeypatch):
    captured = {}

    def fake_json(method, path, payload=None):
        captured.update({"method": method, "path": path, "payload": payload})
        return {"id": "batch_123", "status": "validating"}

    monkeypatch.setattr(batch_api, "_json", fake_json)
    result = batch_api.create_batch("file_123", {"stage": "router"})
    assert result["id"] == "batch_123"
    assert captured == {
        "method": "POST", "path": "/batches",
        "payload": {"input_file_id": "file_123", "endpoint": "/v1/chat/completions",
                    "completion_window": "24h", "metadata": {"stage": "router"}},
    }


def test_collect_downloads_only_completed_batch(monkeypatch, tmp_path):
    monkeypatch.setattr(batch_api, "get_batch", lambda _id: {"status": "in_progress"})
    assert batch_api.collect_if_complete("batch_123", tmp_path / "out.jsonl")["status"] == "in_progress"

    monkeypatch.setattr(batch_api, "get_batch", lambda _id: {"status": "completed", "output_file_id": "file_out"})
    monkeypatch.setattr(batch_api, "download_file", lambda _id: b'{"custom_id":"doc"}\n')
    result = batch_api.collect_if_complete("batch_123", tmp_path / "out.jsonl")
    assert result["status"] == "completed"
    assert json.loads((tmp_path / "out.jsonl").read_text())["custom_id"] == "doc"
