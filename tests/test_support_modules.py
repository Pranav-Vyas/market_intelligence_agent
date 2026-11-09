"""The small helpers that the larger tests only reach indirectly."""

from __future__ import annotations

import json

import pytest

from mia import stubs
from mia.jsonfile import write_json


def test_write_json_creates_folders_and_leaves_no_temp_file(tmp_path):
    target = tmp_path / "a" / "b" / "out.json"
    write_json(target, {"x": 1})
    assert json.loads(target.read_text()) == {"x": 1}
    assert [p.name for p in target.parent.iterdir()] == ["out.json"]


def test_a_failed_write_keeps_the_old_file_and_cleans_up(tmp_path):
    target = tmp_path / "out.json"
    write_json(target, {"x": 1})
    with pytest.raises(TypeError):
        write_json(target, {"x": object()})  # not JSON: fails part way
    assert json.loads(target.read_text()) == {"x": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]


def test_a_stub_is_reported_once_until_reset():
    @stubs.stub
    def fake() -> int:
        return 7

    stubs.reset()
    assert stubs.used() == []
    assert fake() == 7 and fake() == 7
    assert stubs.used() == [f"{__name__}.test_a_stub_is_reported_once_until_reset.<locals>.fake"]
    stubs.reset()
    assert stubs.used() == []
