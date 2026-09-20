"""TypeSafe (Jev) second-stage relevance judgment for the shortlisted papers.

One System One request carries one score question per paper, so the judgment
only runs on the shortlist after embedding rerank. Any failure returns the
papers unchanged so the daily email never depends on this stage.
"""

import json
import os
import urllib.request

from loguru import logger
from omegaconf import DictConfig

from .protocol import Paper

API_URL = "https://api.typesafe.ai/v1/systemone"

DEFAULT_PROFILE = (
    "Program analysis and software security research: dynamic symbolic execution "
    "(DSE), fuzzing and automated test generation, vulnerability discovery, and "
    "the security of LLM-based / agentic AI systems (prompt injection, tool-use "
    "attacks, agent safety)."
)

DEFAULT_LEVELS = [
    "Unrelated to the research profile: different problem domain, no meaningful overlap.",
    "Tangential: shares methods or communities (e.g. LLM applications, code models) but does not address the profile's problems.",
    "Relevant: directly overlaps a profile topic, though it is not the paper's central contribution.",
    "Core: the paper's main contribution is a profile topic - program analysis, DSE, fuzzing, test generation, or LLM/agent security.",
]


def score_papers(
    papers: list[Paper], profile: str, levels: list[str], model: str = "jev-latest", timeout: int = 120
) -> list[float] | None:
    """Score every paper against the profile on the ordered levels.

    Returns one float per paper (a position along the levels, can fall between
    two of them), or None when the API is not configured.
    """
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        logger.warning("TYPESAFE_API_KEY is not set; skipping TypeSafe judgment")
        return None
    state = {
        "profile": profile,
        "papers": [{"title": p.title, "abstract": (p.abstract or "")[:1500]} for p in papers],
    }
    questions = {
        f"paper_{i}": {
            "type": "score",
            "instructions": (
                f"How relevant is `papers[{i}]` to the research interests described in "
                "`profile`? Judge the paper's core contribution, not surface keywords."
            ),
            "criteria": levels,
        }
        for i in range(len(papers))
    }
    body = json.dumps({"state": state, "model": model, "questions": questions}).encode()
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        answers = json.loads(resp.read())["answers"]
    return [float(answers[f"paper_{i}"]["score"]) for i in range(len(papers))]


def refine_top_papers(papers: list[Paper], ts_cfg: DictConfig) -> list[Paper]:
    """Re-sort the shortlist by TypeSafe relevance, dropping unrelated papers."""
    try:
        scores = score_papers(
            papers,
            profile=ts_cfg.profile or DEFAULT_PROFILE,
            levels=list(ts_cfg.levels) if ts_cfg.levels else DEFAULT_LEVELS,
            model=ts_cfg.get("model", "jev-latest"),
        )
    except Exception as e:
        logger.warning(f"TypeSafe judgment failed, keeping embedding order: {e}")
        return papers
    if scores is None:
        return papers
    for p, s in zip(papers, scores):
        p.score = s
    min_score = ts_cfg.get("min_score", 1)
    kept = sorted((p for p in papers if p.score >= min_score), key=lambda p: p.score, reverse=True)
    dropped = len(papers) - len(kept)
    if dropped:
        logger.info(f"TypeSafe dropped {dropped} unrelated paper(s) (score < {min_score})")
    logger.info("TypeSafe relevance order: " + ", ".join(f"{round(p.score, 1)}:{p.title[:50]}" for p in kept))
    return kept
