"""
Table 1 Reproduction: DeepResearchGym Performance.

Compares FlashResearch against GPT-Researcher baseline
at 2-minute and 10-minute time budgets.

Paper configuration:
- Models: o3-mini for planning, gpt-4o-mini for research nodes
- max_depth=10, max_breadth=4-6
- Hard timer interrupt at 120s and 600s
"""

import asyncio
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from langchain_core.messages import HumanMessage

from ..config import AgentConfig, ModelConfig
from ..graph import build_app
from ..memory import apply_memory_seed
from ..baselines import run_sequential_baseline, BaselineResult
from .judge import LLMJudge, JudgeScores
from .datasets import ResearchyQuestion


@dataclass
class Table1Result:
    """Result for a single Table 1 experiment run."""
    question_id: str
    query: str
    system: str                  # "flashresearch" or "baseline"
    time_budget_s: int           # 120 or 600
    elapsed_time: float
    node_count: int
    findings_count: int
    report_length: int
    scores: JudgeScores
    report: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["scores"] = self.scores.to_dict()
        return d


@dataclass
class Table1Summary:
    """Aggregated Table 1 results."""
    system: str
    time_budget_s: int
    num_questions: int
    avg_quality: float
    avg_relevance: float
    avg_faithfulness: float
    avg_overall: float
    avg_node_count: float
    avg_elapsed_time: float
    std_overall: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Paper model configuration
PAPER_MODEL_CONFIG = ModelConfig(
    planner_model="o3-mini-2025-01-31",  # Planning/orchestration
    scorer_model="gpt-4o-mini",
    synthesizer_model="gpt-4o-mini",      # Research nodes
)

# Alternative for users without o3-mini access
FALLBACK_MODEL_CONFIG = ModelConfig(
    planner_model="gpt-4o",
    scorer_model="gpt-4o-mini",
    synthesizer_model="gpt-4o-mini",
)


def get_table1_config(time_budget_s: int, use_paper_models: bool = False) -> AgentConfig:
    """
    Get configuration matching paper's experimental setup.

    Args:
        time_budget_s: Time budget (120 or 600 seconds)
        use_paper_models: Use exact paper models (requires o3-mini access)

    Returns:
        AgentConfig for Table 1 experiments
    """
    return AgentConfig(
        time_budget_s=time_budget_s,
        max_depth=10,               # Paper: max_depth=10
        max_breadth=5,              # Paper: 4-6, using 5
        max_concurrency=8,
        stop_threshold=0.90,
        models=PAPER_MODEL_CONFIG if use_paper_models else FALLBACK_MODEL_CONFIG,
    )


async def run_flashresearch_single(
    question: ResearchyQuestion,
    config: AgentConfig,
) -> Dict[str, Any]:
    """
    Run FlashResearch on a single question with hard timer interrupt.

    Implements the paper's time control: once timer hits,
    trigger synthesis immediately with gathered findings.
    """
    app = build_app()

    state = {
        "messages": [HumanMessage(content=question.query)],
        "root_query": question.query,
        "config": config,
        "start_time": time.time(),
        "time_budget_s": config.time_budget_s,
        "stop": False,
        "max_depth": config.max_depth,
        "max_breadth": config.max_breadth,
        "max_concurrency": config.max_concurrency,
        "pending": [],
        "in_flight": 0,
        "child_tasks": [],
        "findings": [],
        "seen_urls": set(),
        "best_score": 0.0,
        "stop_threshold": config.stop_threshold,
        "tasks_dispatched": 0,
        "tasks_completed": 0,
        "final_report": "",
    }
    state = apply_memory_seed(state, config, question.query)

    try:
        # Run with timeout (hard interrupt)
        out = await asyncio.wait_for(
            app.ainvoke(state),
            timeout=config.time_budget_s + 30  # Small buffer for synthesis
        )
    except asyncio.TimeoutError:
        # If timeout, the graph should have already synthesized
        out = state

    elapsed = time.time() - state["start_time"]

    return {
        "report": out.get("final_report", ""),
        "findings": out.get("findings", []),
        "node_count": out.get("tasks_completed", 0),
        "elapsed_time": elapsed,
        "seen_urls": out.get("seen_urls", set()),
    }


async def run_baseline_single(
    question: ResearchyQuestion,
    config: AgentConfig,
) -> BaselineResult:
    """Run GPT-Researcher style baseline on a single question."""
    return await run_sequential_baseline(question.query, config)


async def run_table1_experiment(
    questions: List[ResearchyQuestion],
    time_budgets: List[int] = [120, 600],
    use_paper_models: bool = False,
    judge_model: str = "gpt-4o",
    run_baseline: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run full Table 1 experiment.

    Args:
        questions: List of research questions
        time_budgets: Time budgets to test (default: [120, 600])
        use_paper_models: Use exact paper model configuration
        judge_model: Model for LLM judge
        run_baseline: Whether to run baseline comparison
        verbose: Print progress

    Returns:
        Dict with all results and summary tables
    """
    judge = LLMJudge(model=judge_model)
    all_results: List[Table1Result] = []

    for time_budget in time_budgets:
        config = get_table1_config(time_budget, use_paper_models)

        if verbose:
            print(f"\n{'='*60}")
            print(f"TIME BUDGET: {time_budget}s ({time_budget//60} minutes)")
            print(f"{'='*60}")

        for i, question in enumerate(questions):
            if verbose:
                print(f"\n[{i+1}/{len(questions)}] {question.id}: {question.query[:50]}...")

            # Run FlashResearch
            if verbose:
                print("  Running FlashResearch...")

            fr_result = await run_flashresearch_single(question, config)
            fr_scores = await judge.score(
                query=question.query,
                report=fr_result["report"],
                findings=fr_result["findings"],
            )

            all_results.append(Table1Result(
                question_id=question.id,
                query=question.query,
                system="flashresearch",
                time_budget_s=time_budget,
                elapsed_time=fr_result["elapsed_time"],
                node_count=fr_result["node_count"],
                findings_count=len(fr_result["findings"]),
                report_length=len(fr_result["report"]),
                scores=fr_scores,
                report=fr_result["report"],
            ))

            if verbose:
                print(f"    FlashResearch: Q={fr_scores.quality:.0f} R={fr_scores.relevance:.0f} F={fr_scores.faithfulness:.0f} Overall={fr_scores.overall:.1f}")

            # Run baseline
            if run_baseline:
                if verbose:
                    print("  Running Baseline...")

                bl_result = await run_baseline_single(question, config)
                bl_scores = await judge.score(
                    query=question.query,
                    report=bl_result.report,
                    findings=bl_result.findings,
                )

                all_results.append(Table1Result(
                    question_id=question.id,
                    query=question.query,
                    system="baseline",
                    time_budget_s=time_budget,
                    elapsed_time=bl_result.elapsed_time,
                    node_count=bl_result.searches_made,
                    findings_count=bl_result.sources_found,
                    report_length=len(bl_result.report),
                    scores=bl_scores,
                    report=bl_result.report,
                ))

                if verbose:
                    print(f"    Baseline: Q={bl_scores.quality:.0f} R={bl_scores.relevance:.0f} F={bl_scores.faithfulness:.0f} Overall={bl_scores.overall:.1f}")

    # Calculate summaries
    summaries = calculate_table1_summaries(all_results)

    return {
        "results": [r.to_dict() for r in all_results],
        "summaries": [s.to_dict() for s in summaries],
        "config": {
            "time_budgets": time_budgets,
            "use_paper_models": use_paper_models,
            "judge_model": judge_model,
            "num_questions": len(questions),
        },
    }


def calculate_table1_summaries(results: List[Table1Result]) -> List[Table1Summary]:
    """Calculate aggregated summaries for Table 1."""
    import statistics

    summaries = []

    # Group by system and time budget
    groups: Dict[tuple, List[Table1Result]] = {}
    for r in results:
        key = (r.system, r.time_budget_s)
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    for (system, time_budget), group_results in groups.items():
        quality_scores = [r.scores.quality for r in group_results]
        relevance_scores = [r.scores.relevance for r in group_results]
        faithfulness_scores = [r.scores.faithfulness for r in group_results]
        overall_scores = [r.scores.overall for r in group_results]
        node_counts = [r.node_count for r in group_results]
        elapsed_times = [r.elapsed_time for r in group_results]

        std_overall = statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0.0

        summaries.append(Table1Summary(
            system=system,
            time_budget_s=time_budget,
            num_questions=len(group_results),
            avg_quality=statistics.mean(quality_scores),
            avg_relevance=statistics.mean(relevance_scores),
            avg_faithfulness=statistics.mean(faithfulness_scores),
            avg_overall=statistics.mean(overall_scores),
            avg_node_count=statistics.mean(node_counts),
            avg_elapsed_time=statistics.mean(elapsed_times),
            std_overall=std_overall,
        ))

    return summaries


def format_table1_results(data: Dict[str, Any]) -> str:
    """Format Table 1 results as a markdown table."""
    summaries = data["summaries"]

    lines = [
        "# Table 1: Performance on DeepResearchGym",
        "",
        "| System | Time Budget | Quality | Relevance | Faithfulness | Overall | Nodes |",
        "|--------|-------------|---------|-----------|--------------|---------|-------|",
    ]

    for s in summaries:
        time_str = f"{s['time_budget_s']//60}min"
        lines.append(
            f"| {s['system']:<14} | {time_str:<11} | "
            f"{s['avg_quality']:.1f} | {s['avg_relevance']:.1f} | "
            f"{s['avg_faithfulness']:.1f} | {s['avg_overall']:.1f}±{s['std_overall']:.1f} | "
            f"{s['avg_node_count']:.1f} |"
        )

    lines.extend([
        "",
        f"_Based on {data['config']['num_questions']} questions._",
    ])

    return "\n".join(lines)
