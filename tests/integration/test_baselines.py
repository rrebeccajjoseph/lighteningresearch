"""Tests for lighteningresearch.baselines module."""

import pytest
from dataclasses import asdict
from io import StringIO
from unittest.mock import patch

from lighteningresearch.baselines import (
    BaselineResult,
    COMMERCIAL_LEADERBOARD,
    print_leaderboard_comparison,
)
from lighteningresearch.config import AgentConfig


class TestBaselineResult:
    """Tests for BaselineResult dataclass."""

    def test_creation(self):
        """Test basic result creation."""
        result = BaselineResult(
            name="sequential_baseline",
            report="This is the research report content.",
            elapsed_time=45.5,
            searches_made=1,
            sources_found=10,
            findings=[
                {"url": "https://example.com", "title": "Test", "content": "Content"}
            ],
        )
        assert result.name == "sequential_baseline"
        assert result.elapsed_time == 45.5
        assert result.searches_made == 1
        assert result.sources_found == 10
        assert len(result.findings) == 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = BaselineResult(
            name="test",
            report="Report",
            elapsed_time=30.0,
            searches_made=5,
            sources_found=20,
            findings=[],
        )
        d = asdict(result)
        assert isinstance(d, dict)
        assert d["name"] == "test"
        assert d["elapsed_time"] == 30.0

    def test_empty_findings(self):
        """Test result with no findings."""
        result = BaselineResult(
            name="test",
            report="",
            elapsed_time=10.0,
            searches_made=1,
            sources_found=0,
            findings=[],
        )
        assert len(result.findings) == 0
        assert result.sources_found == 0


class TestCommercialLeaderboard:
    """Tests for COMMERCIAL_LEADERBOARD constant."""

    def test_leaderboard_not_empty(self):
        """Test that leaderboard has entries."""
        assert len(COMMERCIAL_LEADERBOARD) > 0

    def test_required_systems_present(self):
        """Test that expected systems are in leaderboard."""
        expected_systems = [
            "openai_deep_research",
            "perplexity_research",
            "gpt_researcher",
            "flash_research_paper",
        ]
        for system in expected_systems:
            assert system in COMMERCIAL_LEADERBOARD, f"Missing system: {system}"

    def test_entry_structure(self):
        """Test that each entry has required fields."""
        required_fields = ["name", "race_overall", "citation_accuracy"]

        for key, entry in COMMERCIAL_LEADERBOARD.items():
            for field in required_fields:
                assert field in entry, f"Missing {field} in {key}"

    def test_score_ranges(self):
        """Test that scores are within valid ranges."""
        for key, entry in COMMERCIAL_LEADERBOARD.items():
            assert 0 <= entry["race_overall"] <= 100, f"Invalid RACE score for {key}"
            assert 0 <= entry["citation_accuracy"] <= 100, f"Invalid citation accuracy for {key}"

    def test_gpt_researcher_has_throughput(self):
        """Test that GPT-Researcher has throughput data."""
        gpt_r = COMMERCIAL_LEADERBOARD["gpt_researcher"]
        assert gpt_r["avg_throughput_nodes"] is not None
        assert gpt_r["avg_throughput_nodes"] > 0

    def test_flash_research_paper_scores(self):
        """Test FlashResearch paper reference scores."""
        fr = COMMERCIAL_LEADERBOARD["flash_research_paper"]
        # From paper Table 1
        assert fr["race_overall"] == pytest.approx(41.92, rel=0.1)
        assert fr["citation_accuracy"] == pytest.approx(58.25, rel=0.1)
        assert fr["avg_throughput_nodes"] == pytest.approx(39.30, rel=0.1)


class TestPrintLeaderboardComparison:
    """Tests for print_leaderboard_comparison function."""

    def test_prints_without_error(self, capsys):
        """Test that function prints without error."""
        print_leaderboard_comparison(
            our_race=45.0,
            our_citation_acc=70.0,
            our_throughput=30.0,
        )
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_output_contains_systems(self, capsys):
        """Test that output contains reference systems."""
        print_leaderboard_comparison(
            our_race=45.0,
            our_citation_acc=70.0,
        )
        captured = capsys.readouterr()

        assert "OpenAI Deep Research" in captured.out
        assert "Perplexity" in captured.out
        assert "GPT-Researcher" in captured.out
        assert "FlashResearch" in captured.out

    def test_output_contains_our_scores(self, capsys):
        """Test that output contains our scores."""
        print_leaderboard_comparison(
            our_race=42.5,
            our_citation_acc=68.0,
            our_throughput=35.0,
        )
        captured = capsys.readouterr()

        assert "LightningResearch" in captured.out
        assert "42.5" in captured.out or "42.50" in captured.out

    def test_output_contains_deltas(self, capsys):
        """Test that output contains delta comparison."""
        print_leaderboard_comparison(
            our_race=50.0,
            our_citation_acc=80.0,
        )
        captured = capsys.readouterr()

        assert "Delta" in captured.out

    def test_handles_none_throughput(self, capsys):
        """Test that None throughput is handled gracefully."""
        print_leaderboard_comparison(
            our_race=45.0,
            our_citation_acc=70.0,
            our_throughput=None,
        )
        captured = capsys.readouterr()

        # Should not crash and should show N/A or similar
        assert "N/A" in captured.out or len(captured.out) > 0

    def test_formatting_alignment(self, capsys):
        """Test that output is properly formatted."""
        print_leaderboard_comparison(
            our_race=45.0,
            our_citation_acc=70.0,
            our_throughput=30.0,
        )
        captured = capsys.readouterr()

        # Should have header separators
        assert "=" in captured.out
        assert "-" in captured.out


class TestAgentConfigForBaselines:
    """Tests for AgentConfig as used by baselines."""

    def test_default_config_works(self):
        """Test that default config is valid for baselines."""
        config = AgentConfig()

        # Baselines need these
        assert config.models is not None
        assert config.models.planner_model is not None
        assert config.models.synthesizer_model is not None
        assert config.models.scorer_model is not None
        assert config.search is not None
        assert config.search.max_results_per_search > 0
        assert config.max_breadth > 0

    def test_config_with_custom_models(self):
        """Test config with custom model settings."""
        from lighteningresearch.config import ModelConfig

        config = AgentConfig(
            models=ModelConfig(
                planner_model="gpt-4o",
                synthesizer_model="gpt-4o",
                scorer_model="gpt-4o-mini",
            )
        )

        assert config.models.planner_model == "gpt-4o"
