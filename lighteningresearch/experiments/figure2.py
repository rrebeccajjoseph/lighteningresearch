"""
Figure 2 Reproduction: Quality Trade-off Curves.

Ablation studies showing how quality changes with fixed depth/breadth:
- Figure 2(a): Fix breadth=4, vary depth 1-5
- Figure 2(b): Fix depth=3, vary breadth 1,2,4,8

Records node count as proxy for computational cost.
"""

import asyncio
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.messages import HumanMessage

from ..config import AgentConfig, ModelConfig
from ..graph import build_app
from ..memory import apply_memory_seed
from .judge import LLMJudge, JudgeScores
from .datasets import ResearchyQuestion


@dataclass
class AblationResult:
    """Result from a single ablation run."""
    question_id: str
    query: str
    experiment: str         # "depth" or "breadth"
    fixed_param: int        # The fixed parameter value
    varied_param: int       # The varied parameter value
    depth: int
    breadth: int
    node_count: int         # Key metric for x-axis
    elapsed_time: float
    findings_count: int
    scores: JudgeScores
    report: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["scores"] = self.scores.to_dict()
        return d


@dataclass
class AblationSummary:
    """Aggregated ablation results for plotting."""
    experiment: str
    varied_param: int
    depth: int
    breadth: int
    avg_node_count: float
    avg_quality: float
    avg_relevance: float
    avg_faithfulness: float
    avg_overall: float
    std_overall: float = 0.0
    num_runs: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


async def run_ablation_single(
    question: ResearchyQuestion,
    depth: int,
    breadth: int,
    time_budget_s: int = 300,
    model_config: Optional[ModelConfig] = None,
) -> Dict[str, Any]:
    """
    Run a single ablation experiment with fixed depth/breadth.

    Note: For ablations, we disable adaptive planning and use
    fixed tree parameters.
    """
    config = AgentConfig(
        time_budget_s=time_budget_s,
        max_depth=depth,
        max_breadth=breadth,
        max_concurrency=4,
        stop_threshold=0.95,  # High threshold to not stop early (test full tree)
        models=model_config or ModelConfig(),
    )

    app = build_app()

    state = {
        "messages": [HumanMessage(content=question.query)],
        "root_query": question.query,
        "config": config,
        "start_time": time.time(),
        "time_budget_s": config.time_budget_s,
        "stop": False,
        "max_depth": depth,
        "max_breadth": breadth,
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
        out = await asyncio.wait_for(
            app.ainvoke(state),
            timeout=time_budget_s + 60
        )
    except asyncio.TimeoutError:
        out = state

    elapsed = time.time() - state["start_time"]

    return {
        "report": out.get("final_report", ""),
        "findings": out.get("findings", []),
        "node_count": out.get("tasks_completed", 0),
        "elapsed_time": elapsed,
    }


async def run_depth_ablation(
    questions: List[ResearchyQuestion],
    fixed_breadth: int = 4,
    depth_values: List[int] = [1, 2, 3, 4, 5],
    time_budget_s: int = 300,
    judge_model: str = "gpt-4o",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run depth ablation experiment (Figure 2a).

    Fix breadth at 4, vary depth from 1 to 5.

    Args:
        questions: List of research questions
        fixed_breadth: Fixed breadth value (default: 4)
        depth_values: Depth values to test (default: [1,2,3,4,5])
        time_budget_s: Time budget per run
        judge_model: Model for scoring
        verbose: Print progress

    Returns:
        Dict with results and summaries for plotting
    """
    judge = LLMJudge(model=judge_model)
    all_results: List[AblationResult] = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"DEPTH ABLATION (Fixed breadth={fixed_breadth})")
        print(f"{'='*60}")

    for depth in depth_values:
        if verbose:
            print(f"\n--- Depth = {depth} ---")

        for i, question in enumerate(questions):
            if verbose:
                print(f"  [{i+1}/{len(questions)}] {question.id}...")

            result = await run_ablation_single(
                question=question,
                depth=depth,
                breadth=fixed_breadth,
                time_budget_s=time_budget_s,
            )

            scores = await judge.score(
                query=question.query,
                report=result["report"],
                findings=result["findings"],
            )

            all_results.append(AblationResult(
                question_id=question.id,
                query=question.query,
                experiment="depth",
                fixed_param=fixed_breadth,
                varied_param=depth,
                depth=depth,
                breadth=fixed_breadth,
                node_count=result["node_count"],
                elapsed_time=result["elapsed_time"],
                findings_count=len(result["findings"]),
                scores=scores,
                report=result["report"],
            ))

            if verbose:
                print(f"    Nodes={result['node_count']}, Overall={scores.overall:.1f}")

    summaries = calculate_ablation_summaries(all_results)

    return {
        "experiment": "depth",
        "fixed_param_name": "breadth",
        "fixed_param_value": fixed_breadth,
        "varied_param_name": "depth",
        "varied_values": depth_values,
        "results": [r.to_dict() for r in all_results],
        "summaries": [s.to_dict() for s in summaries],
    }


async def run_breadth_ablation(
    questions: List[ResearchyQuestion],
    fixed_depth: int = 3,
    breadth_values: List[int] = [1, 2, 4, 8],
    time_budget_s: int = 300,
    judge_model: str = "gpt-4o",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run breadth ablation experiment (Figure 2b).

    Fix depth at 3, vary breadth from 1 to 8.

    Args:
        questions: List of research questions
        fixed_depth: Fixed depth value (default: 3)
        breadth_values: Breadth values to test (default: [1,2,4,8])
        time_budget_s: Time budget per run
        judge_model: Model for scoring
        verbose: Print progress

    Returns:
        Dict with results and summaries for plotting
    """
    judge = LLMJudge(model=judge_model)
    all_results: List[AblationResult] = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"BREADTH ABLATION (Fixed depth={fixed_depth})")
        print(f"{'='*60}")

    for breadth in breadth_values:
        if verbose:
            print(f"\n--- Breadth = {breadth} ---")

        for i, question in enumerate(questions):
            if verbose:
                print(f"  [{i+1}/{len(questions)}] {question.id}...")

            result = await run_ablation_single(
                question=question,
                depth=fixed_depth,
                breadth=breadth,
                time_budget_s=time_budget_s,
            )

            scores = await judge.score(
                query=question.query,
                report=result["report"],
                findings=result["findings"],
            )

            all_results.append(AblationResult(
                question_id=question.id,
                query=question.query,
                experiment="breadth",
                fixed_param=fixed_depth,
                varied_param=breadth,
                depth=fixed_depth,
                breadth=breadth,
                node_count=result["node_count"],
                elapsed_time=result["elapsed_time"],
                findings_count=len(result["findings"]),
                scores=scores,
                report=result["report"],
            ))

            if verbose:
                print(f"    Nodes={result['node_count']}, Overall={scores.overall:.1f}")

    summaries = calculate_ablation_summaries(all_results)

    return {
        "experiment": "breadth",
        "fixed_param_name": "depth",
        "fixed_param_value": fixed_depth,
        "varied_param_name": "breadth",
        "varied_values": breadth_values,
        "results": [r.to_dict() for r in all_results],
        "summaries": [s.to_dict() for s in summaries],
    }


def calculate_ablation_summaries(results: List[AblationResult]) -> List[AblationSummary]:
    """Calculate aggregated summaries for plotting."""
    import statistics

    summaries = []

    # Group by (experiment, varied_param)
    groups: Dict[Tuple[str, int], List[AblationResult]] = {}
    for r in results:
        key = (r.experiment, r.varied_param)
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    for (experiment, varied_param), group_results in sorted(groups.items()):
        node_counts = [r.node_count for r in group_results]
        quality_scores = [r.scores.quality for r in group_results]
        relevance_scores = [r.scores.relevance for r in group_results]
        faithfulness_scores = [r.scores.faithfulness for r in group_results]
        overall_scores = [r.scores.overall for r in group_results]

        std_overall = statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0.0

        # Get depth/breadth from first result
        sample = group_results[0]

        summaries.append(AblationSummary(
            experiment=experiment,
            varied_param=varied_param,
            depth=sample.depth,
            breadth=sample.breadth,
            avg_node_count=statistics.mean(node_counts),
            avg_quality=statistics.mean(quality_scores),
            avg_relevance=statistics.mean(relevance_scores),
            avg_faithfulness=statistics.mean(faithfulness_scores),
            avg_overall=statistics.mean(overall_scores),
            std_overall=std_overall,
            num_runs=len(group_results),
        ))

    return summaries


def plot_figure2(
    depth_data: Dict[str, Any],
    breadth_data: Dict[str, Any],
    output_path: str = "figure2.png",
    show: bool = False,
) -> str:
    """
    Generate Figure 2 plots.

    Creates a 2-panel figure:
    - Left (2a): Quality vs Node Count for depth ablation
    - Right (2b): Quality vs Node Count for breadth ablation

    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        return "matplotlib not installed. Install with: pip install matplotlib"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 2(a): Depth ablation
    depth_summaries = depth_data["summaries"]
    node_counts_d = [s["avg_node_count"] for s in depth_summaries]
    quality_d = [s["avg_quality"] for s in depth_summaries]
    relevance_d = [s["avg_relevance"] for s in depth_summaries]
    faithfulness_d = [s["avg_faithfulness"] for s in depth_summaries]

    ax1.plot(node_counts_d, quality_d, 'b-o', label='Quality', linewidth=2, markersize=8)
    ax1.plot(node_counts_d, relevance_d, 'g-s', label='Relevance', linewidth=2, markersize=8)
    ax1.plot(node_counts_d, faithfulness_d, 'r-^', label='Faithfulness', linewidth=2, markersize=8)

    # Add depth labels
    for i, s in enumerate(depth_summaries):
        ax1.annotate(f'd={s["depth"]}', (node_counts_d[i], quality_d[i] + 2),
                     fontsize=9, color='red', ha='center')

    ax1.set_xlabel('Node Count', fontsize=12)
    ax1.set_ylabel('Score (0-100)', fontsize=12)
    ax1.set_title(f'(a) Depth Trade-off (breadth={depth_data["fixed_param_value"]})', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # Plot 2(b): Breadth ablation
    breadth_summaries = breadth_data["summaries"]
    node_counts_b = [s["avg_node_count"] for s in breadth_summaries]
    quality_b = [s["avg_quality"] for s in breadth_summaries]
    relevance_b = [s["avg_relevance"] for s in breadth_summaries]
    faithfulness_b = [s["avg_faithfulness"] for s in breadth_summaries]

    ax2.plot(node_counts_b, quality_b, 'b-o', label='Quality', linewidth=2, markersize=8)
    ax2.plot(node_counts_b, relevance_b, 'g-s', label='Relevance', linewidth=2, markersize=8)
    ax2.plot(node_counts_b, faithfulness_b, 'r-^', label='Faithfulness', linewidth=2, markersize=8)

    # Add breadth labels
    for i, s in enumerate(breadth_summaries):
        ax2.annotate(f'b={s["breadth"]}', (node_counts_b[i], quality_b[i] + 2),
                     fontsize=9, color='red', ha='center')

    ax2.set_xlabel('Node Count', fontsize=12)
    ax2.set_ylabel('Score (0-100)', fontsize=12)
    ax2.set_title(f'(b) Breadth Trade-off (depth={breadth_data["fixed_param_value"]})', fontsize=14)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()

    return output_path


def format_ablation_results(data: Dict[str, Any]) -> str:
    """Format ablation results as markdown table."""
    summaries = data["summaries"]
    experiment = data["experiment"]
    fixed_name = data["fixed_param_name"]
    fixed_value = data["fixed_param_value"]
    varied_name = data["varied_param_name"]

    lines = [
        f"# Figure 2{'(a)' if experiment == 'depth' else '(b)'}: {experiment.title()} Trade-off",
        f"",
        f"Fixed {fixed_name} = {fixed_value}",
        "",
        f"| {varied_name.title()} | Node Count | Quality | Relevance | Faithfulness | Overall |",
        "|--------|------------|---------|-----------|--------------|---------|",
    ]

    for s in summaries:
        lines.append(
            f"| {s['varied_param']} | {s['avg_node_count']:.1f} | "
            f"{s['avg_quality']:.1f} | {s['avg_relevance']:.1f} | "
            f"{s['avg_faithfulness']:.1f} | {s['avg_overall']:.1f}±{s['std_overall']:.1f} |"
        )

    return "\n".join(lines)
