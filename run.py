#!/usr/bin/env python3
"""
LightningResearch CLI - Deep research agent powered by LangGraph.

TIME BUDGET:
    --time, -t {quick,standard,deep}    Time preset (quick=45s, standard=2min, deep=10min)
    --time-seconds SEC                  Custom time budget in seconds

PRESETS:
    --preset, -p {default,academic,news,technical}
                                        Configuration preset

MODELS:
    --model, -m MODEL                   Model for all tasks (e.g., gpt-4o, gpt-4o-mini)
    --planner-model MODEL               Model for query planning
    --synth-model MODEL                 Model for report synthesis

SEARCH:
    --max-results N                     Max results per search (default: 5)
    --include-domains DOMAIN [DOMAIN ...] Only search these domains
    --exclude-domains DOMAIN [DOMAIN ...] Exclude these domains

RESEARCH:
    --max-depth N                       Max research depth (default: 2)
    --max-breadth N                     Max subqueries per level (default: 5)
    --stop-threshold N                  Quality threshold to stop early (default: 0.85)

OUTPUT:
    --output, -o FILE                   Save benchmark results to JSON file
    --eval, -e                          Run RACE and FACT evaluation
    --quiet, -q                         Only print the final report

EXAMPLES:
    python run.py "What is quantum computing?"
    python run.py --time deep "Climate change effects on agriculture"
    python run.py --preset academic "CRISPR gene editing applications"
    python run.py --model gpt-4o --eval "AI safety research"
    python run.py --max-results 10 --include-domains arxiv.org nature.com "Neural networks"
    python run.py --preset technical --quiet "React performance optimization"
"""

import argparse
import json
import time
import asyncio
import os
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from lighteningresearch.graph import build_app
from lighteningresearch.memory import apply_memory_seed
from lighteningresearch.config import (
    AgentConfig,
    ModelConfig,
    SearchConfig,
    LANGSMITH_PROJECT,
    TIME_BUDGETS,
    validate_config,
    academic_config,
    news_config,
    technical_config,
)
from lighteningresearch.evaluation import generate_benchmark_result


# =============================================================================
# Preset configurations
# =============================================================================

PRESETS = {
    "default": AgentConfig,
    "academic": academic_config,
    "news": news_config,
    "technical": technical_config,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="LightningResearch: Deep research agent powered by LangGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py "What is quantum computing?"
  python run.py --time deep "Climate change effects"
  python run.py --preset academic "CRISPR gene editing applications"
  python run.py --model gpt-4o --eval "AI safety research"
  python run.py --max-results 10 --max-depth 3 "Your query"
        """
    )

    # Query
    parser.add_argument(
        "query",
        nargs="*",
        help="Research query (or omit for interactive prompt)"
    )

    # Time options
    time_group = parser.add_argument_group("Time Budget")
    time_group.add_argument(
        "--time", "-t",
        choices=["quick", "standard", "deep"],
        default="quick",
        help="Time budget preset: quick (45s), standard (2min), deep (10min)"
    )
    time_group.add_argument(
        "--time-seconds",
        type=int,
        metavar="SEC",
        help="Custom time budget in seconds (overrides --time)"
    )

    # Preset configurations
    parser.add_argument(
        "--preset", "-p",
        choices=list(PRESETS.keys()),
        default="default",
        help="Configuration preset (default, academic, news, technical)"
    )

    # Model options
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model", "-m",
        default=None,
        help="Model for all tasks (e.g., gpt-4o, gpt-4o-mini)"
    )
    model_group.add_argument(
        "--planner-model",
        default=None,
        help="Model for query planning"
    )
    model_group.add_argument(
        "--synth-model",
        default=None,
        help="Model for report synthesis"
    )

    # Search options
    search_group = parser.add_argument_group("Search Configuration")
    search_group.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Max results per search (default: 5)"
    )
    search_group.add_argument(
        "--include-domains",
        nargs="+",
        metavar="DOMAIN",
        help="Only search these domains (e.g., arxiv.org nature.com)"
    )
    search_group.add_argument(
        "--exclude-domains",
        nargs="+",
        metavar="DOMAIN",
        help="Exclude these domains"
    )

    # Research parameters
    research_group = parser.add_argument_group("Research Parameters")
    research_group.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Max research depth (default: 2)"
    )
    research_group.add_argument(
        "--max-breadth",
        type=int,
        default=None,
        help="Max subqueries per level (default: 5)"
    )
    research_group.add_argument(
        "--stop-threshold",
        type=float,
        default=None,
        help="Quality threshold to stop early (default: 0.85)"
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Save benchmark results to JSON file"
    )
    output_group.add_argument(
        "--eval", "-e",
        action="store_true",
        help="Run RACE and FACT evaluation on the report"
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only print the final report"
    )

    return parser.parse_args()


def build_config(args) -> AgentConfig:
    """Build AgentConfig from CLI arguments."""

    # Start with preset
    if args.preset == "default":
        config = AgentConfig()
    else:
        config = PRESETS[args.preset]()

    # Override time budget
    if args.time_seconds:
        config.time_budget_s = args.time_seconds
    elif args.time:
        config.time_budget_s = TIME_BUDGETS[args.time]

    # Override research parameters
    if args.max_depth is not None:
        config.max_depth = args.max_depth
    if args.max_breadth is not None:
        config.max_breadth = args.max_breadth
    if args.stop_threshold is not None:
        config.stop_threshold = args.stop_threshold

    # Override model settings
    if args.model:
        config.models.planner_model = args.model
        config.models.scorer_model = args.model
        config.models.synthesizer_model = args.model
    if args.planner_model:
        config.models.planner_model = args.planner_model
    if args.synth_model:
        config.models.synthesizer_model = args.synth_model

    # Override search settings
    if args.max_results:
        config.search.max_results_per_search = args.max_results
    if args.include_domains:
        config.search.include_domains = args.include_domains
    if args.exclude_domains:
        config.search.exclude_domains = args.exclude_domains

    return config


async def main():
    args = parse_args()
    validate_config()

    # LangSmith observability
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    # Build configuration
    config = build_config(args)

    # Get query
    if args.query:
        query = " ".join(args.query)
    else:
        query = input("Enter your research query: ").strip()
        if not query:
            print("No query provided. Exiting.")
            return

    if not args.quiet:
        print(f"\n[LightningResearch] Query: {query}")
        print(f"[LightningResearch] Preset: {args.preset}")
        print(f"[LightningResearch] Time budget: {config.time_budget_s}s")
        print(f"[LightningResearch] Model: {config.models.synthesizer_model}")
        if config.search.include_domains:
            print(f"[LightningResearch] Domains: {', '.join(config.search.include_domains)}")
        print()

    app = build_app()

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

    out = await app.ainvoke(state, config={
        "tags": ["lightningresearch", args.preset],
        "metadata": {"time_budget_s": config.time_budget_s},
    })

    # Calculate metrics
    elapsed = time.time() - state["start_time"]
    report = out.get("final_report", "")
    seen_urls = out.get("seen_urls", set())
    tasks_completed = out.get("tasks_completed", 0)
    findings_count = len(out.get("findings", []))

    if not args.quiet:
        # Print throughput stats
        print("\n" + "=" * 60)
        print("THROUGHPUT METRICS")
        print("=" * 60)
        print(f"  Time elapsed:      {elapsed:.1f}s / {config.time_budget_s}s budget")
        print(f"  Tasks dispatched:  {out.get('tasks_dispatched', 0)}")
        print(f"  Tasks completed:   {tasks_completed}")
        print(f"  Findings:          {findings_count}")
        print(f"  Unique sources:    {len(seen_urls)}")
        print(f"  Best score:        {out.get('best_score', 0):.2f}")
        if elapsed > 0:
            print(f"  Efficiency:        {tasks_completed / elapsed:.2f} nodes/sec")
        print("=" * 60)

    # Run evaluation if requested
    if args.eval or args.output:
        if not args.quiet:
            print("\n[Evaluating report quality...]")

        task_id = f"lightning_{uuid.uuid4().hex[:8]}"
        benchmark_result = generate_benchmark_result(
            task_id=task_id,
            query=query,
            report=report,
            source_urls=seen_urls,
            elapsed_time=elapsed,
            tasks_completed=tasks_completed,
            findings_count=findings_count
        )

        if not args.quiet:
            # Print RACE scores
            race = benchmark_result["race_scores"]
            print("\n" + "=" * 60)
            print("RACE EVALUATION (Quality Scores)")
            print("=" * 60)
            print(f"  Comprehensiveness:     {race['comprehensiveness']:.1f}/100")
            print(f"  Depth:                 {race['depth']:.1f}/100")
            print(f"  Instruction Following: {race['instruction_following']:.1f}/100")
            print(f"  Readability:           {race['readability']:.1f}/100")
            print(f"  ─────────────────────────────────")
            print(f"  OVERALL:               {race['overall']:.1f}/100")
            print("=" * 60)

            # Print FACT scores
            fact = benchmark_result["fact_scores"]
            print("\n" + "=" * 60)
            print("FACT EVALUATION (Citation Accuracy)")
            print("=" * 60)
            print(f"  Total citations:       {fact['total_citations']}")
            print(f"  Verified citations:    {fact['verified_citations']}")
            print(f"  Citation accuracy:     {fact['citation_accuracy']:.1f}%")
            print(f"  Citation efficiency:   {fact['citation_efficiency']:.1f} per 1000 words")
            print(f"  Word count:            {fact['word_count']}")
            print("=" * 60)

        # Save to file
        if args.output:
            output_data = {
                **benchmark_result,
                "config": {
                    "preset": args.preset,
                    "time_budget_s": config.time_budget_s,
                    "max_depth": config.max_depth,
                    "max_breadth": config.max_breadth,
                    "model": config.models.synthesizer_model,
                },
                "report": benchmark_result["report"][:500] + "..." if len(benchmark_result["report"]) > 500 else benchmark_result["report"]
            }

            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2, default=str)

            report_file = args.output.replace(".json", "_report.md")
            with open(report_file, "w") as f:
                f.write(f"# Research Report\n\n")
                f.write(f"**Query:** {query}\n\n")
                f.write(f"**Task ID:** {task_id}\n\n")
                f.write(f"**Preset:** {args.preset}\n\n")
                f.write("---\n\n")
                f.write(report)

            if not args.quiet:
                print(f"\n[Results saved to {args.output}]")
                print(f"[Report saved to {report_file}]")

    # Print report
    if not args.quiet:
        print("\n" + "=" * 60)
        print("RESEARCH REPORT")
        print("=" * 60 + "\n")
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
