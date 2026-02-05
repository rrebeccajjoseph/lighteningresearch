#!/usr/bin/env python3
"""
FlashResearch Paper Reproduction Experiments

Reproduce Table 1 (DeepResearchGym Performance) and Figure 2 (Quality Trade-offs)
from the FlashResearch paper.

EXPERIMENTS:
    --table1              Run Table 1 (performance comparison)
    --figure2             Run Figure 2 (depth/breadth ablations)
    --all                 Run all experiments

TABLE 1 OPTIONS:
    --time-budgets SEC    Time budgets in seconds (default: 120,600)
    --run-baseline        Include GPT-Researcher baseline comparison

FIGURE 2 OPTIONS:
    --depth-values N      Depth values to test (default: 1,2,3,4,5)
    --breadth-values N    Breadth values to test (default: 1,2,4,8)
    --fixed-breadth N     Fixed breadth for depth ablation (default: 4)
    --fixed-depth N       Fixed depth for breadth ablation (default: 3)

GENERAL OPTIONS:
    --questions FILE      JSON file with research questions
    --num-questions N     Number of sample questions to use
    --output DIR          Output directory (default: experiment_results/)
    --judge-model MODEL   Model for LLM judge (default: gpt-4o)
    --paper-models        Use exact paper model configuration (requires o3-mini)
    --quick-test          Run quick validation test

EXAMPLES:
    # Run quick test
    python run_experiments.py --quick-test

    # Run Table 1 with 2min and 10min budgets
    python run_experiments.py --table1 --time-budgets 120,600

    # Run Figure 2 ablations
    python run_experiments.py --figure2 --depth-values 1,2,3,4,5

    # Run all experiments on custom questions
    python run_experiments.py --all --questions my_questions.json

    # Full paper reproduction (expensive!)
    python run_experiments.py --all --paper-models --num-questions 100
"""

import argparse
import asyncio
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from lighteningresearch.experiments import (
    ExperimentRunner,
    load_researchy_questions,
    SAMPLE_QUESTIONS,
)
from lighteningresearch.experiments.runner import quick_test
from lighteningresearch.config import validate_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="FlashResearch Paper Reproduction Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Experiment selection
    exp_group = parser.add_argument_group("Experiments")
    exp_group.add_argument(
        "--table1",
        action="store_true",
        help="Run Table 1 (DeepResearchGym performance comparison)"
    )
    exp_group.add_argument(
        "--figure2",
        action="store_true",
        help="Run Figure 2 (depth/breadth ablation studies)"
    )
    exp_group.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments"
    )
    exp_group.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick validation test"
    )

    # Table 1 options
    t1_group = parser.add_argument_group("Table 1 Options")
    t1_group.add_argument(
        "--time-budgets",
        type=str,
        default="120,600",
        help="Comma-separated time budgets in seconds (default: 120,600)"
    )
    t1_group.add_argument(
        "--run-baseline",
        action="store_true",
        default=True,
        help="Include GPT-Researcher baseline comparison"
    )
    t1_group.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline comparison"
    )

    # Figure 2 options
    f2_group = parser.add_argument_group("Figure 2 Options")
    f2_group.add_argument(
        "--depth-values",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated depth values (default: 1,2,3,4,5)"
    )
    f2_group.add_argument(
        "--breadth-values",
        type=str,
        default="1,2,4,8",
        help="Comma-separated breadth values (default: 1,2,4,8)"
    )
    f2_group.add_argument(
        "--fixed-breadth",
        type=int,
        default=4,
        help="Fixed breadth for depth ablation (default: 4)"
    )
    f2_group.add_argument(
        "--fixed-depth",
        type=int,
        default=3,
        help="Fixed depth for breadth ablation (default: 3)"
    )
    f2_group.add_argument(
        "--ablation-time",
        type=int,
        default=300,
        help="Time budget per ablation run in seconds (default: 300)"
    )

    # General options
    gen_group = parser.add_argument_group("General Options")
    gen_group.add_argument(
        "--questions", "-q",
        type=str,
        help="JSON file with research questions"
    )
    gen_group.add_argument(
        "--num-questions", "-n",
        type=int,
        help="Number of sample questions to use"
    )
    gen_group.add_argument(
        "--output", "-o",
        type=str,
        default="experiment_results",
        help="Output directory (default: experiment_results/)"
    )
    gen_group.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        help="Model for LLM judge (default: gpt-4o)"
    )
    gen_group.add_argument(
        "--paper-models",
        action="store_true",
        help="Use exact paper model configuration (requires o3-mini)"
    )
    gen_group.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating matplotlib plots"
    )
    gen_group.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    # Validate environment
    validate_config()

    # Disable LangSmith for experiments
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    # Quick test mode
    if args.quick_test:
        print("\n" + "="*60)
        print("QUICK TEST MODE")
        print("="*60)
        num_q = args.num_questions or 2
        results = await quick_test(num_questions=num_q)
        print("\nQuick test complete!")
        return

    # Load questions
    questions = None
    if args.questions:
        questions = load_researchy_questions(args.questions)
        print(f"Loaded {len(questions)} questions from {args.questions}")
    elif args.num_questions:
        questions = SAMPLE_QUESTIONS[:args.num_questions]
        print(f"Using {len(questions)} sample questions")

    # Parse numeric arguments
    time_budgets = [int(x) for x in args.time_budgets.split(",")]
    depth_values = [int(x) for x in args.depth_values.split(",")]
    breadth_values = [int(x) for x in args.breadth_values.split(",")]

    # Determine which experiments to run
    run_table1 = args.table1 or args.all
    run_figure2 = args.figure2 or args.all

    if not (run_table1 or run_figure2 or args.quick_test):
        print("No experiment specified. Use --table1, --figure2, --all, or --quick-test")
        print("Run with --help for usage information.")
        sys.exit(1)

    # Create runner
    runner = ExperimentRunner(
        output_dir=args.output,
        judge_model=args.judge_model,
        use_paper_models=args.paper_models,
        verbose=not args.quiet,
    )

    results = {}

    # Run Table 1
    if run_table1:
        run_baseline = args.run_baseline and not args.no_baseline
        results["table1"] = await runner.run_table1(
            questions=questions,
            time_budgets=time_budgets,
            run_baseline=run_baseline,
        )

    # Run Figure 2
    if run_figure2:
        results["figure2"] = await runner.run_figure2(
            questions=questions,
            depth_values=depth_values,
            breadth_values=breadth_values,
            fixed_breadth=args.fixed_breadth,
            fixed_depth=args.fixed_depth,
            time_budget_s=args.ablation_time,
            generate_plot=not args.no_plot,
        )

    # Final summary
    if not args.quiet:
        print("\n" + "="*60)
        print("EXPERIMENTS COMPLETE")
        print("="*60)
        print(f"Results saved to: {args.output}/")

        if run_table1 and "table1" in results:
            print("\nTable 1 Summary:")
            for s in results["table1"]["summaries"]:
                print(f"  {s['system']:<15} ({s['time_budget_s']//60}min): Overall={s['avg_overall']:.1f}")

        if run_figure2 and "figure2" in results:
            print("\nFigure 2 Summary:")
            print("  Depth ablation: ", end="")
            depth_overall = [s["avg_overall"] for s in results["figure2"]["depth_ablation"]["summaries"]]
            print(f"Overall range {min(depth_overall):.1f} - {max(depth_overall):.1f}")

            print("  Breadth ablation: ", end="")
            breadth_overall = [s["avg_overall"] for s in results["figure2"]["breadth_ablation"]["summaries"]]
            print(f"Overall range {min(breadth_overall):.1f} - {max(breadth_overall):.1f}")


if __name__ == "__main__":
    asyncio.run(main())
