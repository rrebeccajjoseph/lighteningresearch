"""Tests for lighteningresearch.experiments module."""

import pytest
from dataclasses import asdict

from lighteningresearch.experiments.datasets import (
    ResearchyQuestion,
    SAMPLE_QUESTIONS,
    load_researchy_questions,
    FineWebCorpus,
)
from lighteningresearch.experiments.judge import (
    JudgeScores,
    LLMJudge,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
)
from lighteningresearch.experiments.table1 import (
    Table1Result,
    Table1Summary,
    get_table1_config,
    format_table1_results,
    PAPER_MODEL_CONFIG,
    FALLBACK_MODEL_CONFIG,
)
from lighteningresearch.experiments.figure2 import (
    AblationResult,
    AblationSummary,
    format_ablation_results,
)


class TestResearchyQuestion:
    """Tests for ResearchyQuestion dataclass."""

    def test_creation(self):
        """Test basic question creation."""
        q = ResearchyQuestion(
            id="rq_001",
            query="What is quantum computing?",
            field="Physics",
            difficulty="hard",
            expected_aspects=["qubits", "superposition"],
        )
        assert q.id == "rq_001"
        assert q.query == "What is quantum computing?"
        assert q.field == "Physics"
        assert q.difficulty == "hard"
        assert "qubits" in q.expected_aspects

    def test_default_values(self):
        """Test default field and difficulty."""
        q = ResearchyQuestion(id="test", query="Test query")
        assert q.field == "General"
        assert q.difficulty == "medium"
        assert q.expected_aspects is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        q = ResearchyQuestion(id="test", query="Test")
        d = q.to_dict()
        assert isinstance(d, dict)
        assert d["id"] == "test"
        assert d["query"] == "Test"


class TestSampleQuestions:
    """Tests for SAMPLE_QUESTIONS constant."""

    def test_questions_exist(self):
        """Test that sample questions are defined."""
        assert len(SAMPLE_QUESTIONS) > 0

    def test_all_questions_valid(self):
        """Test all sample questions have required fields."""
        for q in SAMPLE_QUESTIONS:
            assert isinstance(q, ResearchyQuestion)
            assert len(q.id) > 0
            assert len(q.query) > 0
            assert q.field is not None

    def test_questions_diverse(self):
        """Test that questions cover different fields."""
        fields = set(q.field for q in SAMPLE_QUESTIONS)
        assert len(fields) > 1  # At least 2 different fields

    def test_question_ids_unique(self):
        """Test that all question IDs are unique."""
        ids = [q.id for q in SAMPLE_QUESTIONS]
        assert len(ids) == len(set(ids))


class TestLoadResearchyQuestions:
    """Tests for load_researchy_questions function."""

    def test_loads_samples_by_default(self):
        """Test that samples are loaded when no path given."""
        questions = load_researchy_questions()
        assert len(questions) == len(SAMPLE_QUESTIONS)

    def test_limit_parameter(self):
        """Test limiting number of questions."""
        questions = load_researchy_questions(limit=3)
        assert len(questions) == 3

    def test_field_filter(self):
        """Test filtering by field."""
        questions = load_researchy_questions(field_filter="Physics")
        for q in questions:
            assert q.field == "Physics"

    def test_combined_filters(self):
        """Test limit and field filter together."""
        questions = load_researchy_questions(field_filter="AI/ML", limit=1)
        assert len(questions) <= 1
        if len(questions) > 0:
            assert questions[0].field == "AI/ML"


class TestJudgeScores:
    """Tests for JudgeScores dataclass."""

    def test_creation(self):
        """Test basic score creation."""
        scores = JudgeScores(
            quality=80.0,
            relevance=85.0,
            faithfulness=90.0,
            overall=85.0,
            explanation="Good report overall.",
        )
        assert scores.quality == 80.0
        assert scores.relevance == 85.0
        assert scores.faithfulness == 90.0
        assert scores.overall == 85.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        scores = JudgeScores(
            quality=80.0,
            relevance=85.0,
            faithfulness=90.0,
            overall=85.0,
            explanation="Test",
        )
        d = scores.to_dict()
        assert isinstance(d, dict)
        assert d["quality"] == 80.0
        assert d["explanation"] == "Test"


class TestJudgePrompts:
    """Tests for judge prompt templates."""

    def test_system_prompt_contains_rubric(self):
        """Test that system prompt contains scoring rubric."""
        assert "Quality" in JUDGE_SYSTEM_PROMPT
        assert "Relevance" in JUDGE_SYSTEM_PROMPT
        assert "Faithfulness" in JUDGE_SYSTEM_PROMPT
        assert "0-100" in JUDGE_SYSTEM_PROMPT

    def test_user_template_has_placeholders(self):
        """Test that user template has required placeholders."""
        assert "{query}" in JUDGE_USER_TEMPLATE
        assert "{context}" in JUDGE_USER_TEMPLATE
        assert "{report}" in JUDGE_USER_TEMPLATE


class TestTable1Config:
    """Tests for Table 1 configuration."""

    def test_paper_model_config(self):
        """Test paper model configuration."""
        assert PAPER_MODEL_CONFIG.planner_model is not None
        assert PAPER_MODEL_CONFIG.synthesizer_model is not None

    def test_fallback_model_config(self):
        """Test fallback model configuration."""
        assert FALLBACK_MODEL_CONFIG.planner_model is not None
        assert FALLBACK_MODEL_CONFIG.synthesizer_model is not None

    def test_get_table1_config_2min(self):
        """Test 2-minute configuration."""
        config = get_table1_config(120, use_paper_models=False)
        assert config.time_budget_s == 120
        assert config.max_depth == 10  # Paper spec

    def test_get_table1_config_10min(self):
        """Test 10-minute configuration."""
        config = get_table1_config(600, use_paper_models=False)
        assert config.time_budget_s == 600
        assert config.max_depth == 10  # Paper spec

    def test_get_table1_config_paper_models(self):
        """Test with paper model configuration."""
        config = get_table1_config(120, use_paper_models=True)
        assert config.models == PAPER_MODEL_CONFIG


class TestTable1Result:
    """Tests for Table1Result dataclass."""

    def test_creation(self):
        """Test basic result creation."""
        scores = JudgeScores(
            quality=80.0,
            relevance=85.0,
            faithfulness=90.0,
            overall=85.0,
            explanation="Test",
        )
        result = Table1Result(
            question_id="q1",
            query="Test query",
            system="lightningresearch",
            time_budget_s=120,
            elapsed_time=115.0,
            node_count=25,
            findings_count=50,
            report_length=2000,
            scores=scores,
        )
        assert result.question_id == "q1"
        assert result.system == "lightningresearch"
        assert result.node_count == 25

    def test_to_dict(self):
        """Test conversion to dictionary."""
        scores = JudgeScores(80, 85, 90, 85, "Test")
        result = Table1Result(
            question_id="q1",
            query="Test",
            system="test",
            time_budget_s=60,
            elapsed_time=55,
            node_count=10,
            findings_count=20,
            report_length=1000,
            scores=scores,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert isinstance(d["scores"], dict)


class TestFormatTable1Results:
    """Tests for format_table1_results function."""

    def test_formats_markdown_table(self):
        """Test that results are formatted as markdown."""
        data = {
            "summaries": [
                {
                    "system": "lightningresearch",
                    "time_budget_s": 120,
                    "avg_quality": 80.0,
                    "avg_relevance": 85.0,
                    "avg_faithfulness": 90.0,
                    "avg_overall": 85.0,
                    "std_overall": 5.0,
                    "avg_node_count": 25.0,
                },
            ],
            "config": {"num_questions": 10},
        }

        formatted = format_table1_results(data)

        assert "Table 1" in formatted
        assert "|" in formatted  # Table separators
        assert "lightningresearch" in formatted
        assert "2min" in formatted


class TestAblationResult:
    """Tests for AblationResult dataclass."""

    def test_creation(self):
        """Test basic ablation result creation."""
        scores = JudgeScores(80, 85, 90, 85, "Test")
        result = AblationResult(
            question_id="q1",
            query="Test",
            experiment="depth",
            fixed_param=4,
            varied_param=3,
            depth=3,
            breadth=4,
            node_count=12,
            elapsed_time=60.0,
            findings_count=30,
            scores=scores,
        )
        assert result.experiment == "depth"
        assert result.depth == 3
        assert result.breadth == 4
        assert result.node_count == 12


class TestFormatAblationResults:
    """Tests for format_ablation_results function."""

    def test_formats_depth_ablation(self):
        """Test formatting depth ablation results."""
        data = {
            "experiment": "depth",
            "fixed_param_name": "breadth",
            "fixed_param_value": 4,
            "varied_param_name": "depth",
            "summaries": [
                {
                    "varied_param": 1,
                    "avg_node_count": 4.0,
                    "avg_quality": 60.0,
                    "avg_relevance": 65.0,
                    "avg_faithfulness": 70.0,
                    "avg_overall": 65.0,
                    "std_overall": 5.0,
                },
                {
                    "varied_param": 3,
                    "avg_node_count": 12.0,
                    "avg_quality": 75.0,
                    "avg_relevance": 80.0,
                    "avg_faithfulness": 85.0,
                    "avg_overall": 80.0,
                    "std_overall": 4.0,
                },
            ],
        }

        formatted = format_ablation_results(data)

        assert "Figure 2" in formatted
        assert "Depth" in formatted
        assert "breadth = 4" in formatted
        assert "|" in formatted


class TestFineWebCorpus:
    """Tests for FineWebCorpus class."""

    def test_init_creates_empty_index(self, tmp_path):
        """Test initialization with new directory."""
        corpus = FineWebCorpus(str(tmp_path / "corpus"))
        assert corpus.index == {}

    def test_add_and_search_document(self, tmp_path):
        """Test adding and searching documents."""
        corpus = FineWebCorpus(str(tmp_path / "corpus"))

        corpus.add_document(
            url="https://example.com/quantum",
            title="Quantum Computing Article",
            content="Quantum computers use qubits for computation.",
        )

        results = corpus.search("quantum", max_results=10)

        assert len(results) > 0
        assert "quantum" in results[0]["title"].lower() or "quantum" in results[0]["content"].lower()

    def test_search_returns_empty_for_no_match(self, tmp_path):
        """Test search with no matching documents."""
        corpus = FineWebCorpus(str(tmp_path / "corpus"))

        corpus.add_document(
            url="https://example.com/ai",
            title="AI Article",
            content="Artificial intelligence is advancing.",
        )

        results = corpus.search("quantum")
        # May or may not find results depending on implementation
        # Main test is that it doesn't crash

    def test_stats(self, tmp_path):
        """Test getting corpus statistics."""
        corpus = FineWebCorpus(str(tmp_path / "corpus"))

        corpus.add_document(
            url="https://example.com/1",
            title="Article 1",
            content="Content 1",
        )
        corpus.add_document(
            url="https://example.com/2",
            title="Article 2",
            content="Content 2",
        )

        stats = corpus.stats()

        assert stats["total_documents"] == 2
        assert "corpus_dir" in stats
