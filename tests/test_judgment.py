"""Tests for zotero_arxiv_daily.judgment: score_papers, refine_top_papers."""

import json

import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.judgment import score_papers, refine_top_papers
from zotero_arxiv_daily.protocol import Paper


LEVELS = ["Unrelated", "Tangential", "Relevant", "Core"]


def _paper(title="A paper", abstract="An abstract."):
    return Paper(source="arxiv", title=title, authors=["A"], abstract=abstract, url="http://arxiv.org/abs/1")


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


def _fake_urlopen(monkeypatch, scores, captured=None):
    def handler(request, timeout=None):
        if captured is not None:
            captured["body"] = json.loads(request.data)
            captured["headers"] = dict(request.headers)
        payload = {
            "answers": {f"paper_{i}": {"type": "score", "score": s} for i, s in enumerate(scores)}
        }
        return _FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", handler)


def test_score_parsers_and_payload(monkeypatch):
    captured = {}
    _fake_urlopen(monkeypatch, [3.0, 0.5], captured)
    monkeypatch.setenv("TYPESAFE_API_KEY", "apikey_test")

    papers = [_paper("Relevant work"), _paper("Random work")]
    scores = score_papers(papers, profile="fuzzing", levels=LEVELS)

    assert scores == [3.0, 0.5]
    body = captured["body"]
    assert body["model"] == "jev-latest"
    assert body["state"]["profile"] == "fuzzing"
    assert body["state"]["papers"][0]["title"] == "Relevant work"
    assert captured["headers"]["Authorization"] == "Bearer apikey_test"
    question = body["questions"]["paper_1"]
    assert question["type"] == "score"
    assert question["criteria"] == LEVELS


def test_score_papers_without_api_key_returns_none(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert score_papers([_paper()], profile="p", levels=LEVELS) is None


def test_refine_top_papers_sorts_and_drops(monkeypatch):
    _fake_urlopen(monkeypatch, [0.2, 3.0, 1.5])
    monkeypatch.setenv("TYPESAFE_API_KEY", "apikey_test")

    papers = [_paper("noise"), _paper("core"), _paper("tangential")]
    cfg = OmegaConf.create({"profile": None, "levels": None, "min_score": 1, "model": "jev-latest"})
    kept = refine_top_papers(papers, cfg)

    assert [p.title for p in kept] == ["core", "tangential"]
    assert [p.score for p in kept] == [3.0, 1.5]


def test_refine_top_papers_falls_back_on_api_error(monkeypatch):
    def boom(request, timeout=None):
        raise RuntimeError("api down")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    monkeypatch.setenv("TYPESAFE_API_KEY", "apikey_test")

    papers = [_paper("first"), _paper("second")]
    cfg = OmegaConf.create({"profile": None, "levels": None, "min_score": 1, "model": "jev-latest"})
    assert refine_top_papers(papers, cfg) == papers
