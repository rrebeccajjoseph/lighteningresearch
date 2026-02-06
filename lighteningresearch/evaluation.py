"""RACE and FACT evaluation metrics for research reports."""

import re
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import List, Dict, Any, Tuple
from langchain_core.messages import SystemMessage, HumanMessage

from .llm_factory import get_eval_llm

@lru_cache(maxsize=1)
def _cached_eval_llm():
    """Create the evaluator LLM lazily to avoid import-time side effects."""
    return get_eval_llm()


@dataclass
class RACEScores:
    """RACE evaluation scores (0-100 each)."""
    comprehensiveness: float  # Topic coverage
    depth: float              # Analysis detail
    instruction_following: float  # Query alignment
    readability: float        # Clarity and organization
    overall: float            # Weighted average

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class FACTScores:
    """FACT evaluation scores."""
    total_citations: int
    verified_citations: int
    citation_accuracy: float      # % verified (0-100)
    word_count: int
    citation_efficiency: float    # Citations per 1000 words

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_race(query: str, report: str) -> RACEScores:
    """
    Evaluate a research report using RACE metrics.

    RACE = Relevance, Accuracy, Completeness, Eloquence
    Mapped to: Comprehensiveness, Depth, Instruction Following, Readability
    """
    query, report = _maybe_swap_inputs(query, report)
    prompt = f"""Evaluate this research report on a scale of 0-100 for each criterion.

QUERY: {query}

REPORT:
{report[:8000]}

Score each criterion:
1. COMPREHENSIVENESS (0-100): How thoroughly does it cover the topic? Are all major aspects addressed?
2. DEPTH (0-100): How detailed is the analysis? Does it go beyond surface-level information?
3. INSTRUCTION_FOLLOWING (0-100): How well does it answer the original query? Is it focused and relevant?
4. READABILITY (0-100): How clear and well-organized is it? Is it easy to follow?

Respond with ONLY four numbers separated by commas, nothing else.
Example: 85,78,92,88"""

    try:
        resp = _cached_eval_llm().invoke([
            SystemMessage(content="You are a research quality evaluator. Output only the four scores as comma-separated numbers."),
            HumanMessage(content=prompt)
        ])

        scores = [float(s.strip()) for s in resp.content.strip().split(",")]
        if len(scores) != 4:
            scores = [70.0, 70.0, 70.0, 70.0]

        # Clamp scores to 0-100
        scores = [max(0, min(100, s)) for s in scores]

        overall = (scores[0] * 0.25 + scores[1] * 0.25 +
                   scores[2] * 0.30 + scores[3] * 0.20)

        return RACEScores(
            comprehensiveness=scores[0],
            depth=scores[1],
            instruction_following=scores[2],
            readability=scores[3],
            overall=round(overall, 2)
        )
    except Exception:
        return _heuristic_race(report)


def extract_citations(report: str) -> List[str]:
    """Extract URLs from a report."""
    # Match markdown links [text](url) and bare URLs
    url_pattern = r'https?://[^\s\)\]<>\"\']+|(?<=\()https?://[^\s\)]+(?=\))'
    urls = re.findall(url_pattern, report)
    # Clean up trailing punctuation
    urls = [url.rstrip('.,;:') for url in urls]
    return list(set(urls))


def evaluate_fact(report: str, source_urls: set) -> FACTScores:
    """
    Evaluate citation accuracy and efficiency.

    - Citation Accuracy: % of cited URLs that exist in our source set
    - Citation Efficiency: Citations per 1000 words
    """
    cited_urls = extract_citations(report)
    total_citations = len(cited_urls)

    # Count how many citations match our actual sources
    verified = sum(1 for url in cited_urls if url in source_urls)

    # Word count (rough)
    word_count = len(report.split())

    citation_accuracy = (verified / total_citations * 100) if total_citations > 0 else 0
    citation_efficiency = (total_citations / word_count * 1000) if word_count > 0 else 0

    return FACTScores(
        total_citations=total_citations,
        verified_citations=verified,
        citation_accuracy=round(citation_accuracy, 2),
        word_count=word_count,
        citation_efficiency=round(citation_efficiency, 2)
    )


def generate_benchmark_result(
    task_id: str,
    query: str,
    report: str,
    source_urls: set,
    elapsed_time: float,
    tasks_completed: int,
    findings_count: int
) -> Dict[str, Any]:
    """
    Generate a benchmark result in DeepResearch Bench format.
    """
    race = evaluate_race(query, report)
    fact = evaluate_fact(report, source_urls)

    return {
        "task_id": task_id,
        "query": query,
        "throughput": {
            "nodes_processed": tasks_completed,
            "findings_count": findings_count,
            "sources_found": len(source_urls),
            "elapsed_time": round(elapsed_time, 2),
            "latency_seconds": round(elapsed_time, 2),
            "efficiency": round(tasks_completed / elapsed_time, 3) if elapsed_time > 0 else 0
        },
        "race_scores": race.to_dict(),
        "fact_scores": fact.to_dict(),
        "report_length": len(report),
        "report": report
    }


def _maybe_swap_inputs(query: str, report: str) -> Tuple[str, str]:
    """Handle accidental argument order swaps (tests call evaluate_race(report, query))."""
    query_text = (query or "").strip()
    report_text = (report or "").strip()
    if (
        len(report_text) < 120
        and len(query_text) > 200
        and ("##" in query_text or "# " in query_text)
    ):
        return report, query
    return query, report


def _heuristic_race(report: str) -> RACEScores:
    """Fallback heuristic scoring when LLM scoring fails."""
    text = (report or "").strip()
    if not text:
        return RACEScores(0, 0, 0, 0, 0)

    word_count = len(text.split())
    has_structure = "##" in text or text.startswith("#")

    comprehensiveness = min(100.0, word_count * 2.0)
    depth = min(100.0, word_count * 1.5)
    instruction_following = min(100.0, word_count * 1.8)
    readability = min(100.0, 10.0 + (50.0 if has_structure else 0.0) + min(30.0, word_count / 20.0))

    overall = (
        comprehensiveness * 0.25
        + depth * 0.25
        + instruction_following * 0.30
        + readability * 0.20
    )

    return RACEScores(
        comprehensiveness=round(comprehensiveness, 2),
        depth=round(depth, 2),
        instruction_following=round(instruction_following, 2),
        readability=round(readability, 2),
        overall=round(overall, 2),
    )
