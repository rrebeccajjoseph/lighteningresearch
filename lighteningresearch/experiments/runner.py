"""
Unified experiment runner for reproducing paper results.

Provides a single interface to run:
- Table 1: DeepResearchGym performance
- Figure 2: Quality trade-off curves
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .datasets import load_researchy_questions, ResearchyQuestion, SAMPLE_QUESTIONS
from .table1 import run_table1_experiment, format_table1_results
from .figure2 import (
    run_depth_ablation,
    run_breadth_ablation,
    plot_figure2,
    format_ablation_results,
)
from .judge import LLMJudge


class ExperimentRunner:
    """
    Unified runner for paper reproduction experiments.

    Usage:
        runner = ExperimentRunner(output_dir="experiments/")

        # Run Table 1
        results = await runner.run_table1(questions, time_budgets=[120, 600])

        # Run Figure 2
        results = await runner.run_figure2(questions)

        # Run all experiments
        results = await runner.run_all(questions)
    """

    def __init__(
        self,
        output_dir: str = "experiment_results",
        judge_model: str = "gpt-4o",
        use_paper_models: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize experiment runner.

        Args:
            output_dir: Directory for saving results
            judge_model: Model to use for LLM judge
            use_paper_models: Use exact paper model configuration
            verbose: Print progress
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.judge_model = judge_model
        self.use_paper_models = use_paper_models
        self.verbose = verbose

        # Sub-directories
        (self.output_dir / "table1").mkdir(exist_ok=True)
        (self.output_dir / "figure2").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)

    async def run_table1(
        self,
        questions: Optional[List[ResearchyQuestion]] = None,
        time_budgets: List[int] = [120, 600],
        run_baseline: bool = True,
    ) -> Dict[str, Any]:
        """
        Run Table 1 experiment (DeepResearchGym performance).

        Args:
            questions: Questions to test (uses samples if None)
            time_budgets: Time budgets in seconds [120, 600]
            run_baseline: Run GPT-Researcher baseline comparison

        Returns:
            Dict with results and formatted table
        """
        if questions is None:
            questions = SAMPLE_QUESTIONS[:5]  # Use first 5 samples

        if self.verbose:
            print("\n" + "="*70)
            print("RUNNING TABLE 1: DeepResearchGym Performance")
            print("="*70)
            print(f"Questions: {len(questions)}")
            print(f"Time budgets: {time_budgets}")
            print(f"Run baseline: {run_baseline}")

        results = await run_table1_experiment(
            questions=questions,
            time_budgets=time_budgets,
            use_paper_models=self.use_paper_models,
            judge_model=self.judge_model,
            run_baseline=run_baseline,
            verbose=self.verbose,
        )

        # Format and save
        formatted = format_table1_results(results)
        results["formatted_table"] = formatted

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / "table1" / f"table1_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        table_file = self.output_dir / "table1" / f"table1_{timestamp}.md"
        with open(table_file, "w") as f:
            f.write(formatted)

        if self.verbose:
            print(f"\n{formatted}")
            print(f"\nResults saved to: {results_file}")
            print(f"Table saved to: {table_file}")

        return results

    async def run_figure2(
        self,
        questions: Optional[List[ResearchyQuestion]] = None,
        depth_values: List[int] = [1, 2, 3, 4, 5],
        breadth_values: List[int] = [1, 2, 4, 8],
        fixed_breadth: int = 4,
        fixed_depth: int = 3,
        time_budget_s: int = 300,
        generate_plot: bool = True,
    ) -> Dict[str, Any]:
        """
        Run Figure 2 experiments (quality trade-off curves).

        Args:
            questions: Questions to test (uses samples if None)
            depth_values: Depths to test for ablation [1,2,3,4,5]
            breadth_values: Breadths to test for ablation [1,2,4,8]
            fixed_breadth: Fixed breadth for depth ablation
            fixed_depth: Fixed depth for breadth ablation
            time_budget_s: Time budget per run
            generate_plot: Generate matplotlib plot

        Returns:
            Dict with depth and breadth ablation results
        """
        if questions is None:
            questions = SAMPLE_QUESTIONS[:3]  # Use first 3 for ablations

        if self.verbose:
            print("\n" + "="*70)
            print("RUNNING FIGURE 2: Quality Trade-off Curves")
            print("="*70)
            print(f"Questions: {len(questions)}")
            print(f"Depth values: {depth_values}")
            print(f"Breadth values: {breadth_values}")

        # Run depth ablation (Figure 2a)
        depth_results = await run_depth_ablation(
            questions=questions,
            fixed_breadth=fixed_breadth,
            depth_values=depth_values,
            time_budget_s=time_budget_s,
            judge_model=self.judge_model,
            verbose=self.verbose,
        )

        # Run breadth ablation (Figure 2b)
        breadth_results = await run_breadth_ablation(
            questions=questions,
            fixed_depth=fixed_depth,
            breadth_values=breadth_values,
            time_budget_s=time_budget_s,
            judge_model=self.judge_model,
            verbose=self.verbose,
        )

        combined = {
            "depth_ablation": depth_results,
            "breadth_ablation": breadth_results,
            "config": {
                "depth_values": depth_values,
                "breadth_values": breadth_values,
                "fixed_breadth": fixed_breadth,
                "fixed_depth": fixed_depth,
                "time_budget_s": time_budget_s,
                "num_questions": len(questions),
            },
        }

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / "figure2" / f"figure2_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(combined, f, indent=2)

        # Save formatted tables
        depth_table = format_ablation_results(depth_results)
        breadth_table = format_ablation_results(breadth_results)

        table_file = self.output_dir / "figure2" / f"figure2_tables_{timestamp}.md"
        with open(table_file, "w") as f:
            f.write(depth_table)
            f.write("\n\n")
            f.write(breadth_table)

        # Generate plot
        plot_file = None
        if generate_plot:
            plot_file = str(self.output_dir / "figure2" / f"figure2_{timestamp}.png")
            try:
                plot_figure2(depth_results, breadth_results, plot_file)
                combined["plot_file"] = plot_file
                if self.verbose:
                    print(f"\nPlot saved to: {plot_file}")
            except Exception as e:
                if self.verbose:
                    print(f"\nCould not generate plot: {e}")

        if self.verbose:
            print(f"\n{depth_table}")
            print(f"\n{breadth_table}")
            print(f"\nResults saved to: {results_file}")

        return combined

    async def run_all(
        self,
        questions: Optional[List[ResearchyQuestion]] = None,
        table1_time_budgets: List[int] = [120, 600],
        figure2_depth_values: List[int] = [1, 2, 3, 4, 5],
        figure2_breadth_values: List[int] = [1, 2, 4, 8],
    ) -> Dict[str, Any]:
        """
        Run all paper reproduction experiments.

        Args:
            questions: Questions to use for all experiments
            table1_time_budgets: Time budgets for Table 1
            figure2_depth_values: Depth values for Figure 2
            figure2_breadth_values: Breadth values for Figure 2

        Returns:
            Dict with all experiment results
        """
        if questions is None:
            questions = SAMPLE_QUESTIONS

        if self.verbose:
            print("\n" + "#"*70)
            print("# RUNNING ALL PAPER REPRODUCTION EXPERIMENTS")
            print("#"*70)
            print(f"Total questions: {len(questions)}")

        results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "judge_model": self.judge_model,
                "use_paper_models": self.use_paper_models,
                "num_questions": len(questions),
            },
        }

        # Table 1
        results["table1"] = await self.run_table1(
            questions=questions[:5],  # Use subset for Table 1
            time_budgets=table1_time_budgets,
        )

        # Figure 2
        results["figure2"] = await self.run_figure2(
            questions=questions[:3],  # Use smaller subset for ablations
            depth_values=figure2_depth_values,
            breadth_values=figure2_breadth_values,
        )

        # Save combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_file = self.output_dir / f"all_experiments_{timestamp}.json"
        with open(combined_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        if self.verbose:
            print("\n" + "#"*70)
            print("# ALL EXPERIMENTS COMPLETE")
            print("#"*70)
            print(f"Combined results saved to: {combined_file}")

        return results


async def quick_test(num_questions: int = 2) -> Dict[str, Any]:
    """
    Quick test run with minimal questions for validation.

    Args:
        num_questions: Number of questions to test

    Returns:
        Test results
    """
    runner = ExperimentRunner(
        output_dir="quick_test_results",
        verbose=True,
    )

    questions = SAMPLE_QUESTIONS[:num_questions]

    print("\n" + "="*50)
    print("QUICK TEST MODE")
    print("="*50)

    # Quick Table 1 (short time budget)
    table1 = await runner.run_table1(
        questions=questions,
        time_budgets=[60],  # 1 minute only
        run_baseline=False,  # Skip baseline for speed
    )

    # Quick Figure 2 (fewer points)
    figure2 = await runner.run_figure2(
        questions=questions[:1],
        depth_values=[1, 2],
        breadth_values=[1, 2],
        time_budget_s=60,
        generate_plot=True,
    )

    return {
        "table1": table1,
        "figure2": figure2,
    }
