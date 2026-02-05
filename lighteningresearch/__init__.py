"""
LightningResearch: A deep research agent powered by LangGraph.

This package provides a multi-agent research system that:
- Accepts natural language research queries
- Performs parallel web searches using Tavily
- Scores and filters results by relevance
- Synthesizes findings into structured reports
- Evaluates quality using RACE and FACT metrics

Example:
    from lighteningresearch import build_app, AgentConfig

    # Use a preset configuration
    config = AgentConfig.deep()

    # Or customize
    config = AgentConfig(
        time_budget_s=300,
        max_depth=3,
        models=ModelConfig(synthesizer_model="gpt-4o"),
    )
"""

from .models import FRState, Task, Finding
from .graph import build_app
from .config import (
    AgentConfig,
    ModelConfig,
    PromptConfig,
    ReportConfig,
    ReportSection,
    SearchConfig,
    validate_config,
    TIME_BUDGETS,
    academic_config,
    news_config,
    technical_config,
)
from .evaluation import (
    evaluate_race,
    evaluate_fact,
    generate_benchmark_result,
    RACEScores,
    FACTScores,
)
from .baselines import (
    run_sequential_baseline,
    run_parallel_no_scoring,
    run_no_adaptive_depth,
    print_leaderboard_comparison,
    COMMERCIAL_LEADERBOARD,
)
from .cache import (
    SearchCache,
    CachedSearchTool,
    create_reproducible_corpus,
)

# Experiments (for paper reproduction)
from .experiments import (
    ExperimentRunner,
    LLMJudge,
    JudgeScores,
    load_researchy_questions,
    ResearchyQuestion,
    SAMPLE_QUESTIONS,
    run_table1_experiment,
    run_depth_ablation,
    run_breadth_ablation,
    plot_figure2,
)

__version__ = "0.2.0"
__all__ = [
    # Core
    "FRState",
    "Task",
    "Finding",
    "build_app",

    # Configuration
    "AgentConfig",
    "ModelConfig",
    "PromptConfig",
    "ReportConfig",
    "ReportSection",
    "SearchConfig",
    "validate_config",
    "TIME_BUDGETS",

    # Presets
    "academic_config",
    "news_config",
    "technical_config",

    # Evaluation
    "evaluate_race",
    "evaluate_fact",
    "generate_benchmark_result",
    "RACEScores",
    "FACTScores",

    # Baselines & Benchmarking
    "run_sequential_baseline",
    "run_parallel_no_scoring",
    "run_no_adaptive_depth",
    "print_leaderboard_comparison",
    "COMMERCIAL_LEADERBOARD",

    # Reproducibility
    "SearchCache",
    "CachedSearchTool",
    "create_reproducible_corpus",

    # Experiments
    "ExperimentRunner",
    "LLMJudge",
    "JudgeScores",
    "load_researchy_questions",
    "ResearchyQuestion",
    "SAMPLE_QUESTIONS",
    "run_table1_experiment",
    "run_depth_ablation",
    "run_breadth_ablation",
    "plot_figure2",
]
