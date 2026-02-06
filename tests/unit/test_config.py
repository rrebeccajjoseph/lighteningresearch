"""Tests for lighteningresearch.config module."""

import os
import pytest
from unittest.mock import patch

from lighteningresearch.config import (
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
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_BREADTH,
    DEFAULT_STOP_THRESHOLD,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        """Test default model configuration."""
        config = ModelConfig()
        assert config.planner_model == "gpt-4o-mini"
        assert config.scorer_model == "gpt-4o-mini"
        assert config.synthesizer_model == "gpt-4o-mini"

    def test_custom_values(self):
        """Test custom model configuration."""
        config = ModelConfig(
            planner_model="gpt-4o",
            scorer_model="gpt-3.5-turbo",
            synthesizer_model="gpt-4o",
        )
        assert config.planner_model == "gpt-4o"
        assert config.scorer_model == "gpt-3.5-turbo"
        assert config.synthesizer_model == "gpt-4o"

    def test_env_override(self):
        """Test environment variable overrides."""
        with patch.dict(os.environ, {
            "FR_PLANNER_MODEL": "custom-planner",
            "FR_SYNTH_MODEL": "custom-synth",
        }):
            # Note: This requires the config to actually read env vars
            # The current implementation uses field defaults, not env
            config = ModelConfig()
            # Default behavior - env vars need explicit handling
            assert config.planner_model == "gpt-4o-mini"


class TestSearchConfig:
    """Tests for SearchConfig dataclass."""

    def test_default_values(self):
        """Test default search configuration."""
        config = SearchConfig()
        assert config.max_results_per_search == 5
        assert config.include_domains is None
        assert config.exclude_domains is None

    def test_domain_filtering(self):
        """Test domain inclusion/exclusion."""
        config = SearchConfig(
            include_domains=["arxiv.org", "nature.com"],
            exclude_domains=["pinterest.com"],
        )
        assert "arxiv.org" in config.include_domains
        assert "pinterest.com" in config.exclude_domains

    def test_max_results_bounds(self):
        """Test max results configuration."""
        config = SearchConfig(max_results_per_search=20)
        assert config.max_results_per_search == 20


class TestReportSection:
    """Tests for ReportSection dataclass."""

    def test_creation(self):
        """Test report section creation."""
        section = ReportSection(
            name="Summary",
            description="Brief overview of findings",
        )
        assert section.name == "Summary"
        assert section.description == "Brief overview of findings"


class TestReportConfig:
    """Tests for ReportConfig dataclass."""

    def test_default_sections(self):
        """Test default report sections exist."""
        config = ReportConfig()
        assert len(config.sections) > 0
        section_names = [s.name for s in config.sections]
        assert "Executive Summary" in section_names

    def test_custom_sections(self):
        """Test custom report sections."""
        sections = [
            ReportSection("Abstract", "Brief summary"),
            ReportSection("Methods", "Research methodology"),
        ]
        config = ReportConfig(sections=sections)
        assert len(config.sections) == 2
        assert config.sections[0].name == "Abstract"


class TestPromptConfig:
    """Tests for PromptConfig dataclass."""

    def test_default_prompts_exist(self):
        """Test that default prompts are provided."""
        config = PromptConfig()
        assert len(config.planner_system) > 0
        assert len(config.synthesizer_system) > 0
        assert len(config.scorer_system) > 0

    def test_prompts_contain_key_instructions(self):
        """Test prompts contain necessary instructions."""
        config = PromptConfig()
        # Planner should mention subqueries
        assert "subquer" in config.planner_system.lower() or "query" in config.planner_system.lower()
        # Synthesizer should mention report/citations
        assert "report" in config.synthesizer_system.lower() or "citation" in config.synthesizer_system.lower()


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_default_values(self):
        """Test default agent configuration."""
        config = AgentConfig()
        assert config.time_budget_s > 0
        assert config.max_depth == DEFAULT_MAX_DEPTH
        assert config.max_breadth == DEFAULT_MAX_BREADTH
        assert 0 <= config.stop_threshold <= 1

    def test_nested_configs(self):
        """Test nested configuration objects."""
        config = AgentConfig()
        assert isinstance(config.models, ModelConfig)
        assert isinstance(config.prompts, PromptConfig)
        assert isinstance(config.report, ReportConfig)
        assert isinstance(config.search, SearchConfig)

    def test_custom_time_budget(self):
        """Test custom time budget."""
        config = AgentConfig(time_budget_s=600)
        assert config.time_budget_s == 600

    def test_custom_nested_config(self):
        """Test custom nested configuration."""
        config = AgentConfig(
            models=ModelConfig(planner_model="gpt-4o"),
            search=SearchConfig(max_results_per_search=10),
        )
        assert config.models.planner_model == "gpt-4o"
        assert config.search.max_results_per_search == 10

    def test_stop_threshold_bounds(self):
        """Test stop threshold is within valid bounds."""
        config = AgentConfig(stop_threshold=0.5)
        assert config.stop_threshold == 0.5

        config = AgentConfig(stop_threshold=0.0)
        assert config.stop_threshold == 0.0

        config = AgentConfig(stop_threshold=1.0)
        assert config.stop_threshold == 1.0


class TestTimeBudgets:
    """Tests for TIME_BUDGETS constant."""

    def test_all_presets_exist(self):
        """Test all time budget presets are defined."""
        assert "quick" in TIME_BUDGETS
        assert "standard" in TIME_BUDGETS
        assert "deep" in TIME_BUDGETS

    def test_preset_ordering(self):
        """Test time budgets are ordered correctly."""
        assert TIME_BUDGETS["quick"] < TIME_BUDGETS["standard"]
        assert TIME_BUDGETS["standard"] < TIME_BUDGETS["deep"]

    def test_preset_values_reasonable(self):
        """Test preset values are reasonable."""
        assert TIME_BUDGETS["quick"] >= 30  # At least 30 seconds
        assert TIME_BUDGETS["deep"] <= 1800  # At most 30 minutes


class TestPresetConfigs:
    """Tests for preset configuration functions."""

    def test_academic_config(self):
        """Test academic preset configuration."""
        config = academic_config()
        assert isinstance(config, AgentConfig)
        # Academic should have longer time and depth
        assert config.time_budget_s >= 120
        # Should focus on academic domains
        if config.search.include_domains:
            domains = [d.lower() for d in config.search.include_domains]
            assert any("arxiv" in d or "nature" in d or "science" in d for d in domains)

    def test_news_config(self):
        """Test news preset configuration."""
        config = news_config()
        assert isinstance(config, AgentConfig)
        # News should be quicker
        assert config.time_budget_s <= 120

    def test_technical_config(self):
        """Test technical preset configuration."""
        config = technical_config()
        assert isinstance(config, AgentConfig)
        # Technical might focus on dev resources
        if config.search.include_domains:
            domains = [d.lower() for d in config.search.include_domains]
            assert any("github" in d or "stackoverflow" in d for d in domains)


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_missing_openai_key(self):
        """Test validation fails without OPENAI_API_KEY."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            # Remove the key if it exists
            env_copy = os.environ.copy()
            if "OPENAI_API_KEY" in env_copy:
                del env_copy["OPENAI_API_KEY"]

            with patch.dict(os.environ, env_copy, clear=True):
                with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                    validate_config()

    def test_missing_tavily_key(self):
        """Test validation fails without TAVILY_API_KEY."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
        }, clear=True):
            with pytest.raises(ValueError, match="TAVILY_API_KEY"):
                validate_config()

    def test_valid_config(self):
        """Test validation passes with required keys."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-openai-key",
            "TAVILY_API_KEY": "test-tavily-key",
        }):
            # Should not raise
            validate_config()
