"""
Experimental pipeline for reproducing FlashResearch paper results.

Provides:
- Table 1: DeepResearchGym performance comparison
- Figure 2: Quality trade-off curves (depth/breadth ablations)
- LLM Judge for Quality/Relevance/Faithfulness scoring
"""

from .judge import LLMJudge, JudgeScores, score_report
from .datasets import (
    load_researchy_questions,
    ResearchyQuestion,
    SAMPLE_QUESTIONS,
)
from .table1 import (
    run_table1_experiment,
    Table1Result,
    format_table1_results,
)
from .figure2 import (
    run_depth_ablation,
    run_breadth_ablation,
    AblationResult,
    plot_figure2,
    format_ablation_results,
)
from .runner import ExperimentRunner

__all__ = [
    # Judge
    "LLMJudge",
    "JudgeScores",
    "score_report",

    # Datasets
    "load_researchy_questions",
    "ResearchyQuestion",
    "SAMPLE_QUESTIONS",

    # Table 1
    "run_table1_experiment",
    "Table1Result",
    "format_table1_results",

    # Figure 2
    "run_depth_ablation",
    "run_breadth_ablation",
    "AblationResult",
    "plot_figure2",
    "format_ablation_results",

    # Runner
    "ExperimentRunner",
]
