"""Preset configurations for common research modes."""

from ..config import AgentConfig, ReportConfig, ReportSection, SearchConfig


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
