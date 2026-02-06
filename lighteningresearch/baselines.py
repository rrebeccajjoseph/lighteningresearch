"""
Baseline implementations and reference scores for benchmarking.

Provides:
1. Ablated implementations to prove orchestration improvements
2. Commercial leaderboard reference scores for comparison
3. Reproducibility utilities
"""

import time
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, SystemMessage

from .tools import make_search_tool
from .config import AgentConfig
from .llm_factory import get_research_llm, get_fast_llm


@dataclass
class BaselineResult:
    """Result from a baseline run."""
    name: str
    report: str
    elapsed_time: float
    searches_made: int
    sources_found: int
    findings: List[Dict[str, Any]]


# =============================================================================
# Commercial Leaderboard Reference Scores
# =============================================================================

# Reference scores from DeepResearch Bench paper
COMMERCIAL_LEADERBOARD = {
    "openai_deep_research": {
        "name": "OpenAI Deep Research",
        "race_overall": 46.45,
        "citation_accuracy": 75.01,
        "avg_throughput_nodes": None,  # N/A in paper
    },
    "perplexity_research": {
        "name": "Perplexity Research",
        "race_overall": 40.46,
        "citation_accuracy": 82.63,
        "avg_throughput_nodes": None,  # N/A in paper
    },
    "gpt_researcher": {
        "name": "GPT-Researcher (Baseline)",
        "race_overall": 41.15,
        "citation_accuracy": 65.58,
        "avg_throughput_nodes": 23.12,
    },
    "flash_research_paper": {
        "name": "FlashResearch (Paper)",
        "race_overall": 41.92,
        "citation_accuracy": 58.25,
        "avg_throughput_nodes": 39.30,
    },
}


def print_leaderboard_comparison(
    our_race: float,
    our_citation_acc: float,
    our_throughput: Optional[float] = None,
):
    """Print formatted leaderboard comparison table."""
    print("\n" + "=" * 80)
    print("LEADERBOARD COMPARISON (vs DeepResearch Bench)")
    print("=" * 80)
    print(f"{'System':<30} {'RACE':>12} {'Citation':>12} {'Throughput':>12}")
    print("-" * 80)

    # Print reference systems
    for ref in COMMERCIAL_LEADERBOARD.values():
        throughput_str = f"{ref['avg_throughput_nodes']:.2f}" if ref['avg_throughput_nodes'] else "N/A"
        print(f"{ref['name']:<30} {ref['race_overall']:>12.2f} {ref['citation_accuracy']:>12.2f} {throughput_str:>12}")

    print("-" * 80)

    # Print our results
    throughput_str = f"{our_throughput:.2f}" if our_throughput else "N/A"
    print(f"{'LightningResearch (Ours)':<30} {our_race:>12.2f} {our_citation_acc:>12.2f} {throughput_str:>12}")

    # Print deltas vs GPT-Researcher baseline
    baseline = COMMERCIAL_LEADERBOARD["gpt_researcher"]
    race_delta = our_race - baseline["race_overall"]
    citation_delta = our_citation_acc - baseline["citation_accuracy"]

    print("-" * 80)
    print(f"{'Delta vs GPT-Researcher':<30} {race_delta:>+12.2f} {citation_delta:>+12.2f}", end="")
    if our_throughput and baseline["avg_throughput_nodes"]:
        throughput_delta = our_throughput - baseline["avg_throughput_nodes"]
        print(f" {throughput_delta:>+12.2f}")
    else:
        print()

    print("=" * 80)


# =============================================================================
# Baseline 1: Sequential (like basic GPT-Researcher)
# =============================================================================

async def run_sequential_baseline(
    query: str,
    config: AgentConfig,
) -> BaselineResult:
    """
    Simple sequential baseline (no parallelism, no scoring).

    This represents naive single-query approach.
    """
    start_time = time.time()
    llm = get_research_llm(model=config.models.synthesizer_model, temperature=0.2)
    search_tool = make_search_tool(max_results=config.search.max_results_per_search)

    results = await search_tool.ainvoke(query)

    if isinstance(results, dict):
        results = results.get("results") or results.get("data") or []

    findings = []
    evidence_text = []

    for r in (results or []):
        if isinstance(r, dict):
            url = r.get("url", "")
            title = r.get("title", "")
            content = r.get("content", r.get("snippet", ""))[:500]
            if url and content:
                findings.append({"url": url, "title": title, "content": content})
                evidence_text.append(f"- {title} ({url})\n  {content}")

    prompt = f"""Write a research report answering: {query}

Evidence:
{chr(10).join(evidence_text)}

Include executive summary, key findings, and conclusion."""

    resp = llm.invoke([HumanMessage(content=prompt)])

    return BaselineResult(
        name="sequential_baseline",
        report=resp.content,
        elapsed_time=time.time() - start_time,
        searches_made=1,
        sources_found=len(findings),
        findings=findings,
    )


# =============================================================================
# Baseline 2: Parallel No Scoring (Ablation)
# =============================================================================

async def run_parallel_no_scoring(
    query: str,
    config: AgentConfig,
) -> BaselineResult:
    """
    Parallel search WITHOUT scoring.

    Tests whether scoring/filtering adds value.
    """
    start_time = time.time()
    llm = get_research_llm(model=config.models.planner_model, temperature=0.3)
    search_tool = make_search_tool(max_results=config.search.max_results_per_search)

    # Generate subqueries
    plan_resp = llm.invoke([HumanMessage(
        content=f"Generate {config.max_breadth} subqueries for: {query}\nReturn newline-separated list."
    )])
    subqueries = [q.strip() for q in plan_resp.content.split("\n") if q.strip()][:config.max_breadth]

    # Parallel search
    async def search_one(q):
        try:
            return await search_tool.ainvoke(q)
        except:
            return []

    all_results = await asyncio.gather(*[search_one(q) for q in subqueries])

    findings = []
    seen = set()

    for results in all_results:
        if isinstance(results, dict):
            results = results.get("results") or []
        for r in (results or []):
            if isinstance(r, dict):
                url = r.get("url", "")
                if url and url not in seen:
                    seen.add(url)
                    findings.append({
                        "url": url,
                        "title": r.get("title", ""),
                        "content": r.get("content", "")[:500],
                    })

    evidence = "\n".join(f"- {f['title']} ({f['url']})\n  {f['content']}" for f in findings[:12])

    synth_llm = get_research_llm(model=config.models.synthesizer_model, temperature=0.2)
    resp = synth_llm.invoke([HumanMessage(
        content=f"Write research report for: {query}\n\nEvidence:\n{evidence}"
    )])

    return BaselineResult(
        name="parallel_no_scoring",
        report=resp.content,
        elapsed_time=time.time() - start_time,
        searches_made=len(subqueries),
        sources_found=len(findings),
        findings=findings,
    )


# =============================================================================
# Baseline 3: No Adaptive Depth (Ablation)
# =============================================================================

async def run_no_adaptive_depth(
    query: str,
    config: AgentConfig,
) -> BaselineResult:
    """
    Parallel + scoring but NO child task spawning.

    Tests whether adaptive depth adds value.
    """
    start_time = time.time()
    llm = get_research_llm(model=config.models.planner_model, temperature=0.3)
    scorer = get_fast_llm(model=config.models.scorer_model, temperature=0.1)
    search_tool = make_search_tool(max_results=config.search.max_results_per_search)

    # Generate subqueries
    plan_resp = llm.invoke([HumanMessage(
        content=f"Generate {config.max_breadth} subqueries for: {query}\nReturn newline-separated list."
    )])
    subqueries = [q.strip() for q in plan_resp.content.split("\n") if q.strip()][:config.max_breadth]

    # Parallel search
    async def search_one(q):
        try:
            return await search_tool.ainvoke(q)
        except:
            return []

    all_results = await asyncio.gather(*[search_one(q) for q in subqueries])

    findings = []
    seen = set()

    for results in all_results:
        if isinstance(results, dict):
            results = results.get("results") or []
        for r in (results or []):
            if isinstance(r, dict):
                url = r.get("url", "")
                title = r.get("title", "")
                content = r.get("content", "")[:500]

                if url and url not in seen and content:
                    seen.add(url)

                    # Score
                    try:
                        score_resp = scorer.invoke([
                            SystemMessage(content="Score 0.0-1.0. Output ONLY number."),
                            HumanMessage(content=f"Query: {query}\nTitle: {title}\nContent: {content}")
                        ])
                        score = float(score_resp.content.strip())
                    except:
                        score = 0.3

                    findings.append({"url": url, "title": title, "content": content, "score": score})

    # Sort by score
    findings.sort(key=lambda x: x.get("score", 0), reverse=True)
    top = findings[:12]

    evidence = "\n".join(f"- {f['title']} ({f['url']})\n  {f['content']}" for f in top)

    synth_llm = get_research_llm(model=config.models.synthesizer_model, temperature=0.2)
    resp = synth_llm.invoke([HumanMessage(
        content=f"Write research report for: {query}\n\nEvidence:\n{evidence}"
    )])

    return BaselineResult(
        name="no_adaptive_depth",
        report=resp.content,
        elapsed_time=time.time() - start_time,
        searches_made=len(subqueries),
        sources_found=len(findings),
        findings=findings,
    )


