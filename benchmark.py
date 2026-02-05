#!/usr/bin/env python3
"""
LightningResearch Benchmark Runner

Features:
- Run benchmarks on multiple tasks
- Compare against baselines (ablations)
- Compare against commercial leaderboard
- Output DeepResearch Bench compatible results

Usage:
      python benchmark.py sample_tasks.json \
    --time standard \
    --compare-baselines \
    --show-leaderboard \
    --output benchmark_results/
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from lighteningresearch.graph import build_app
from lighteningresearch.memory import apply_memory_seed
from lighteningresearch.config import (
    AgentConfig,
    DEFAULT_STOP_THRESHOLD,
    DEFAULT_MAX_BREADTH,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_CONCURRENCY,
    TIME_BUDGETS,
    validate_config,
)
from lighteningresearch.evaluation import generate_benchmark_result
from lighteningresearch.baselines import (
    run_sequential_baseline,
    run_parallel_no_scoring,
    run_no_adaptive_depth,
    print_leaderboard_comparison,
    COMMERCIAL_LEADERBOARD,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="LightningResearch Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark.py sample_tasks.json
    python benchmark.py tasks.json --time deep --output results/
    python benchmark.py tasks.json --compare-baselines
    python benchmark.py tasks.json --show-leaderboard
        """
    )
    parser.add_argument(
        "tasks_file",
        help="JSON file with research tasks"
    )
    parser.add_argument(
        "--output", "-o",
        default="benchmark_output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--time",
        choices=["quick", "standard", "deep"],
        default="standard",
        help="Time budget per task"
    )
    parser.add_argument(
        "--time-seconds",
        type=int,
        help="Custom time budget in seconds"
    )
    parser.add_argument(
        "--compare-baselines",
        action="store_true",
        help="Run ablation baselines for comparison"
    )
    parser.add_argument(
        "--show-leaderboard",
        action="store_true",
        help="Show comparison against commercial leaderboard"
    )
    return parser.parse_args()


async def run_single_task(
    app,
    task: Dict[str, Any],
    config: AgentConfig,
) -> Dict[str, Any]:
    """Run a single research task and return benchmark result."""

    task_id = task.get("id", f"task_{time.time()}")
    query = task.get("query", "")

    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"Query: {query[:80]}...")
    print(f"Time budget: {config.time_budget_s}s")
    print('='*60)

    state = {
        "messages": [HumanMessage(content=query)],
        "root_query": query,
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
    state = apply_memory_seed(state, config, query)

    try:
        out = await app.ainvoke(state, config={
            "tags": ["benchmark", task_id],
        })

        elapsed = time.time() - state["start_time"]
        report = out.get("final_report", "")
        seen_urls = out.get("seen_urls", set())
        tasks_completed = out.get("tasks_completed", 0)
        findings_count = len(out.get("findings", []))

        result = generate_benchmark_result(
            task_id=task_id,
            query=query,
            report=report,
            source_urls=seen_urls,
            elapsed_time=elapsed,
            tasks_completed=tasks_completed,
            findings_count=findings_count
        )

        result["field"] = task.get("field", "General")
        result["status"] = "completed"

        print(f"  Completed in {elapsed:.1f}s")
        print(f"  RACE Overall: {result['race_scores']['overall']:.1f}")
        print(f"  Citation Accuracy: {result['fact_scores']['citation_accuracy']:.1f}%")
        print(f"  Nodes: {tasks_completed}")

        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "task_id": task_id,
            "query": query,
            "status": "failed",
            "error": str(e)
        }


async def run_baseline_comparison(
    query: str,
    config: AgentConfig,
) -> Dict[str, Any]:
    """Run all baselines on a single query for comparison."""

    print("\n" + "=" * 60)
    print("RUNNING BASELINE COMPARISONS")
    print("=" * 60)

    results = {}

    # Sequential baseline
    print("\n[1/3] Sequential baseline (no parallelism)...")
    seq_result = await run_sequential_baseline(query, config)
    results["sequential"] = {
        "elapsed": seq_result.elapsed_time,
        "sources": seq_result.sources_found,
        "searches": seq_result.searches_made,
    }
    print(f"      Time: {seq_result.elapsed_time:.1f}s, Sources: {seq_result.sources_found}")

    # Parallel no scoring
    print("\n[2/3] Parallel without scoring (ablation)...")
    pns_result = await run_parallel_no_scoring(query, config)
    results["parallel_no_scoring"] = {
        "elapsed": pns_result.elapsed_time,
        "sources": pns_result.sources_found,
        "searches": pns_result.searches_made,
    }
    print(f"      Time: {pns_result.elapsed_time:.1f}s, Sources: {pns_result.sources_found}")

    # No adaptive depth
    print("\n[3/3] No adaptive depth (ablation)...")
    nad_result = await run_no_adaptive_depth(query, config)
    results["no_adaptive_depth"] = {
        "elapsed": nad_result.elapsed_time,
        "sources": nad_result.sources_found,
        "searches": nad_result.searches_made,
    }
    print(f"      Time: {nad_result.elapsed_time:.1f}s, Sources: {nad_result.sources_found}")

    return results


async def main():
    args = parse_args()
    validate_config()

    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    # Load tasks
    with open(args.tasks_file) as f:
        data = json.load(f)

    tasks = data.get("tasks", data) if isinstance(data, dict) else data
    if not isinstance(tasks, list):
        tasks = [tasks]

    time_budget = args.time_seconds if args.time_seconds else TIME_BUDGETS[args.time]
    config = AgentConfig(time_budget_s=time_budget)

    print(f"\n{'#'*60}")
    print(f"# LightningResearch Benchmark")
    print(f"# Tasks: {len(tasks)}")
    print(f"# Time budget per task: {time_budget}s")
    print(f"# Output: {args.output}/")
    print(f"{'#'*60}")

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(f"{args.output}/reports", exist_ok=True)

    app = build_app()

    # Run all tasks
    results: List[Dict[str, Any]] = []
    start_time = time.time()

    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}]", end="")
        result = await run_single_task(app, task, config)
        results.append(result)

        if result.get("status") == "completed":
            task_id = result["task_id"]
            report_path = f"{args.output}/reports/{task_id}_report.md"
            with open(report_path, "w") as f:
                f.write(f"# {task.get('query', 'Research Report')}\n\n")
                f.write(result.get("report", ""))

    total_time = time.time() - start_time

    # Calculate aggregate metrics
    completed = [r for r in results if r.get("status") == "completed"]

    if completed:
        avg_race = sum(r["race_scores"]["overall"] for r in completed) / len(completed)
        avg_citation_acc = sum(r["fact_scores"]["citation_accuracy"] for r in completed) / len(completed)
        total_nodes = sum(r["throughput"]["nodes_processed"] for r in completed)
        total_findings = sum(r["throughput"]["findings_count"] for r in completed)
        avg_nodes = total_nodes / len(completed)
    else:
        avg_race = avg_citation_acc = total_nodes = total_findings = avg_nodes = 0

    # Run baseline comparison if requested
    baseline_results = None
    if args.compare_baselines and tasks:
        baseline_results = await run_baseline_comparison(tasks[0]["query"], config)

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print('='*60)
    print(f"  Total time:           {total_time:.1f}s")
    print(f"  Tasks completed:      {len(completed)}/{len(tasks)}")
    print(f"  Avg RACE Overall:     {avg_race:.2f}")
    print(f"  Avg Citation Acc:     {avg_citation_acc:.2f}%")
    print(f"  Total nodes:          {total_nodes}")
    print(f"  Avg nodes/task:       {avg_nodes:.2f}")
    print('='*60)

    # Show leaderboard comparison
    if args.show_leaderboard:
        print_leaderboard_comparison(
            our_race=avg_race,
            our_citation_acc=avg_citation_acc,
            our_throughput=avg_nodes,
        )

    # Build output
    aggregate = {
        "total_tasks": len(tasks),
        "completed_tasks": len(completed),
        "failed_tasks": len(tasks) - len(completed),
        "total_time_seconds": round(total_time, 2),
        "time_budget_per_task": time_budget,
        "avg_race_overall": round(avg_race, 2),
        "avg_citation_accuracy": round(avg_citation_acc, 2),
        "total_nodes_processed": total_nodes,
        "total_findings": total_findings,
        "avg_nodes_per_task": round(avg_nodes, 2),
    }

    benchmark_output = {
        "benchmark": "LightningResearch",
        "version": "0.2.0",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "time_budget_seconds": time_budget,
            "max_depth": config.max_depth,
            "max_breadth": config.max_breadth,
            "stop_threshold": config.stop_threshold,
        },
        "aggregate_metrics": aggregate,
        "leaderboard_reference": COMMERCIAL_LEADERBOARD,
        "baseline_comparison": baseline_results,
        "results": [
            {k: v for k, v in r.items() if k != "report"}
            for r in results
        ]
    }

    output_file = f"{args.output}/benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(benchmark_output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
