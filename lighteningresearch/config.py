"""
Configuration for LightningResearch agent.

All settings can be customized via:
1. Environment variables
2. AgentConfig dataclass
3. Runtime overrides
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List

# =============================================================================
# API Keys
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "lightningresearch")


def validate_config():
    """Validate required environment variables are set."""
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


# =============================================================================
# Time Budget Presets
# =============================================================================

TIME_BUDGETS = {
    "quick": 45,      # Fast results
    "standard": 120,  # 2 minutes
    "deep": 600,      # 10 minutes - thorough research
}


# =============================================================================
# Default Values (can be overridden via environment)
# =============================================================================

DEFAULT_TIME_BUDGET_S = float(os.getenv("FR_TIME_BUDGET_S", "45"))
DEFAULT_STOP_THRESHOLD = float(os.getenv("FR_STOP_THRESHOLD", "0.85"))
DEFAULT_MAX_BREADTH = int(os.getenv("FR_MAX_BREADTH", "5"))
DEFAULT_MAX_DEPTH = int(os.getenv("FR_MAX_DEPTH", "2"))
DEFAULT_MAX_CONCURRENCY = int(os.getenv("FR_MAX_CONCURRENCY", "8"))
DEFAULT_MAX_RESULTS_PER_SEARCH = int(os.getenv("FR_MAX_RESULTS", "5"))
DEFAULT_RETRY_MAX_ATTEMPTS = int(os.getenv("FR_RETRY_MAX_ATTEMPTS", "3"))
DEFAULT_RETRY_INITIAL_BACKOFF_S = float(os.getenv("FR_RETRY_INITIAL_BACKOFF_S", "1.0"))
DEFAULT_RETRY_MAX_BACKOFF_S = float(os.getenv("FR_RETRY_MAX_BACKOFF_S", "8.0"))


def _getenv_bool(name: str, default: str = "false") -> bool:
    val = os.getenv(name, default).strip().lower()
    return val in {"1", "true", "yes", "y", "on"}


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for LLM models used in the agent."""

    # Model for generating subqueries
    planner_model: str = os.getenv("FR_PLANNER_MODEL", "gpt-4o-mini")
    planner_temperature: float = float(os.getenv("FR_PLANNER_TEMP", "0.3"))

    # Model for scoring relevance
    scorer_model: str = os.getenv("FR_SCORER_MODEL", "gpt-4o-mini")
    scorer_temperature: float = float(os.getenv("FR_SCORER_TEMP", "0.1"))

    # Model for synthesizing reports
    synthesizer_model: str = os.getenv("FR_SYNTH_MODEL", "gpt-4o-mini")
    synthesizer_temperature: float = float(os.getenv("FR_SYNTH_TEMP", "0.2"))

    # Model for RACE evaluation
    evaluator_model: str = os.getenv("FR_EVAL_MODEL", "gpt-4o-mini")
    evaluator_temperature: float = float(os.getenv("FR_EVAL_TEMP", "0.1"))


# =============================================================================
# Prompt Templates
# =============================================================================

@dataclass
class PromptConfig:
    """Customizable prompts for each agent node."""

    planner_system: str = """You are a research planner. Generate diverse, specific subqueries
that together will comprehensively answer the root query. Each subquery should explore
a different angle or aspect of the topic."""

    planner_template: str = """Generate {max_breadth} diverse subqueries to research:
Root query: {query}
Return as a plain newline-separated list. No numbering or bullets."""

    scorer_system: str = """Score the usefulness of this content for answering the research query.
Output ONLY a number between 0.0 and 1.0, where:
- 0.0 = completely irrelevant
- 0.5 = somewhat relevant
- 1.0 = highly relevant and authoritative"""

    scorer_template: str = """Root query: {root_query}
Title: {title}
Content: {content}"""

    synthesizer_system: str = """You are a research synthesizer. Write comprehensive,
well-structured reports that directly answer the query using the provided evidence.
Always cite sources inline using markdown links."""

    synthesizer_template: str = """Write a comprehensive research report answering: {query}

{report_structure}

---
Evidence to use (cite these sources):
{evidence}

Important: Ground all claims in the provided evidence. Use inline citations as [Title](URL)."""

    child_task_template: str = """Find additional evidence about: {topic}
Focus on: {focus}
Related to the main query: {root_query}"""


# =============================================================================
# Report Structure Configuration
# =============================================================================

@dataclass
class ReportSection:
    """A section in the research report."""
    name: str
    instruction: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.instruction and self.description:
            self.instruction = self.description
        if not self.description and self.instruction:
            self.description = self.instruction


@dataclass
class ReportConfig:
    """Customizable report structure."""

    sections: List[ReportSection] = field(default_factory=lambda: [
        ReportSection(
            name="Executive Summary",
            instruction="Provide a 2-3 sentence overview of the key findings."
        ),
        ReportSection(
            name="Key Findings",
            instruction="Present the main discoveries as bullet points. Cite sources inline using [Source Title](URL) format."
        ),
        ReportSection(
            name="Analysis",
            instruction="Synthesize the evidence, noting any conflicting information or areas of uncertainty."
        ),
        ReportSection(
            name="Conclusion",
            instruction="Summarize the implications and any gaps in the current evidence."
        ),
    ])

    def to_prompt(self) -> str:
        """Convert report structure to prompt instructions."""
        lines = ["Structure your report as follows:\n"]
        for section in self.sections:
            lines.append(f"## {section.name}")
            lines.append(section.instruction)
            lines.append("")
        return "\n".join(lines)


# =============================================================================
# Search Configuration
# =============================================================================

@dataclass
class SearchConfig:
    """Configuration for web search behavior."""

    max_results_per_search: int = DEFAULT_MAX_RESULTS_PER_SEARCH
    include_domains: Optional[List[str]] = None  # e.g., ["arxiv.org", "nature.com"]
    exclude_domains: Optional[List[str]] = None  # e.g., ["pinterest.com"]
    search_depth: str = "basic"  # "basic" or "advanced"
    include_answer: bool = False
    include_raw_content: bool = False
    cache_enabled: bool = _getenv_bool("FR_SEARCH_CACHE_ENABLED", "false")
    cache_dir: str = os.getenv("FR_SEARCH_CACHE_DIR", ".search_cache")


# =============================================================================
# Memory Configuration
# =============================================================================

@dataclass
class MemoryConfig:
    """Configuration for lightweight cross-run memory."""

    enabled: bool = _getenv_bool("FR_MEMORY_ENABLED", "false")
    path: str = os.getenv("FR_MEMORY_PATH", ".lightning_memory.json")
    min_score: float = float(os.getenv("FR_MEMORY_MIN_SCORE", "0.7"))
    max_seed_findings: int = int(os.getenv("FR_MEMORY_MAX_SEED", "5"))
    max_sources: int = int(os.getenv("FR_MEMORY_MAX_SOURCES", "500"))
    allow_early_stop: bool = _getenv_bool("FR_MEMORY_ALLOW_EARLY_STOP", "false")


# =============================================================================
# Main Agent Configuration
# =============================================================================

@dataclass
class AgentConfig:
    """Complete configuration for the LightningResearch agent."""

    # Time and stopping
    time_budget_s: float = DEFAULT_TIME_BUDGET_S
    stop_threshold: float = DEFAULT_STOP_THRESHOLD

    # Research parameters
    max_breadth: int = DEFAULT_MAX_BREADTH
    max_depth: int = DEFAULT_MAX_DEPTH
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY

    # Retry behavior (for networked tools like search)
    retry_max_attempts: int = DEFAULT_RETRY_MAX_ATTEMPTS
    retry_initial_backoff_s: float = DEFAULT_RETRY_INITIAL_BACKOFF_S
    retry_max_backoff_s: float = DEFAULT_RETRY_MAX_BACKOFF_S

    # Scoring thresholds
    child_task_threshold: float = 0.7  # Min score to spawn child task
    top_findings_for_report: int = 12  # Number of findings for synthesis

    # Sub-configurations
    models: ModelConfig = field(default_factory=ModelConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create config from environment variables."""
        return cls(
            time_budget_s=DEFAULT_TIME_BUDGET_S,
            stop_threshold=DEFAULT_STOP_THRESHOLD,
            max_breadth=DEFAULT_MAX_BREADTH,
            max_depth=DEFAULT_MAX_DEPTH,
            max_concurrency=DEFAULT_MAX_CONCURRENCY,
        )

    @classmethod
    def quick(cls) -> "AgentConfig":
        """Preset for quick research (45s)."""
        return cls(time_budget_s=45, max_depth=1, max_breadth=3)

    @classmethod
    def standard(cls) -> "AgentConfig":
        """Preset for standard research (2min)."""
        return cls(time_budget_s=120, max_depth=2, max_breadth=5)

    @classmethod
    def deep(cls) -> "AgentConfig":
        """Preset for deep research (10min)."""
        return cls(time_budget_s=600, max_depth=3, max_breadth=8)


# =============================================================================
# Example Custom Configurations
# =============================================================================

def academic_config() -> AgentConfig:
    """Configuration optimized for academic research."""
    return AgentConfig(
        time_budget_s=300,
        max_depth=3,
        max_breadth=5,
        search=SearchConfig(
            max_results_per_search=8,
            include_domains=["arxiv.org", "scholar.google.com", "nature.com", "science.org"],
        ),
        report=ReportConfig(sections=[
            ReportSection("Abstract", "Provide a brief abstract of the research findings."),
            ReportSection("Background", "Summarize the relevant background and context."),
            ReportSection("Methodology", "Describe how the research was conducted."),
            ReportSection("Findings", "Present the key findings with citations."),
            ReportSection("Discussion", "Analyze the implications and limitations."),
            ReportSection("References", "List all cited sources."),
        ]),
    )


def news_config() -> AgentConfig:
    """Configuration optimized for news/current events."""
    return AgentConfig(
        time_budget_s=60,
        max_depth=1,
        max_breadth=8,
        stop_threshold=0.75,
        search=SearchConfig(
            max_results_per_search=10,
            exclude_domains=["pinterest.com", "facebook.com"],
        ),
        report=ReportConfig(sections=[
            ReportSection("Summary", "What happened in 2-3 sentences."),
            ReportSection("Key Facts", "Bullet points of confirmed facts with sources."),
            ReportSection("Context", "Background information for understanding."),
            ReportSection("Sources", "List of news sources consulted."),
        ]),
    )


def technical_config() -> AgentConfig:
    """Configuration optimized for technical/developer research."""
    return AgentConfig(
        time_budget_s=180,
        max_depth=2,
        max_breadth=6,
        search=SearchConfig(
            max_results_per_search=8,
            include_domains=["github.com", "stackoverflow.com", "docs.python.org"],
        ),
        report=ReportConfig(sections=[
            ReportSection("Overview", "Brief summary of the solution/approach."),
            ReportSection("Implementation", "Code examples and technical details."),
            ReportSection("Alternatives", "Other approaches considered."),
            ReportSection("Resources", "Links to documentation and examples."),
        ]),
    )
