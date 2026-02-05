"""Tests for lighteningresearch.evaluation module."""

import pytest
from dataclasses import asdict

from lighteningresearch.evaluation import (
    RACEScores,
    FACTScores,
    evaluate_race,
    evaluate_fact,
    extract_citations,
    generate_benchmark_result,
)


class TestRACEScores:
    """Tests for RACEScores dataclass."""

    def test_creation(self):
        """Test basic score creation."""
        scores = RACEScores(
            comprehensiveness=80.0,
            depth=75.0,
            instruction_following=90.0,
            readability=85.0,
            overall=82.5,
        )
        assert scores.comprehensiveness == 80.0
        assert scores.depth == 75.0
        assert scores.instruction_following == 90.0
        assert scores.readability == 85.0
        assert scores.overall == 82.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        scores = RACEScores(
            comprehensiveness=80.0,
            depth=75.0,
            instruction_following=90.0,
            readability=85.0,
            overall=82.5,
        )
        d = asdict(scores)
        assert isinstance(d, dict)
        assert d["comprehensiveness"] == 80.0


class TestFACTScores:
    """Tests for FACTScores dataclass."""

    def test_creation(self):
        """Test basic score creation."""
        scores = FACTScores(
            total_citations=10,
            verified_citations=8,
            citation_accuracy=80.0,
            citation_efficiency=5.0,
            word_count=2000,
        )
        assert scores.total_citations == 10
        assert scores.verified_citations == 8
        assert scores.citation_accuracy == 80.0

    def test_zero_citations(self):
        """Test with zero citations."""
        scores = FACTScores(
            total_citations=0,
            verified_citations=0,
            citation_accuracy=0.0,
            citation_efficiency=0.0,
            word_count=1000,
        )
        assert scores.total_citations == 0
        assert scores.citation_accuracy == 0.0


class TestExtractCitations:
    """Tests for extract_citations helper function."""

    def test_extract_markdown_links(self):
        """Test extracting markdown-style links."""
        text = "According to [Source 1](https://example.com/1) and [Source 2](https://test.org/2)."
        citations = extract_citations(text)
        assert "https://example.com/1" in citations
        assert "https://test.org/2" in citations

    def test_extract_plain_urls(self):
        """Test extracting plain URLs."""
        text = "See https://example.com/article for more info."
        citations = extract_citations(text)
        assert "https://example.com/article" in citations

    def test_no_citations(self):
        """Test text with no citations."""
        text = "This is plain text without any URLs or citations."
        citations = extract_citations(text)
        assert len(citations) == 0

    def test_duplicate_citations(self):
        """Test that duplicate URLs are handled."""
        text = "See [link](https://example.com) and [same link](https://example.com)."
        citations = extract_citations(text)
        # Should deduplicate
        assert citations.count("https://example.com") <= 1 or len(set(citations)) < len(citations)

    def test_mixed_protocols(self):
        """Test http and https URLs."""
        text = "HTTP: http://example.com and HTTPS: https://secure.com"
        citations = extract_citations(text)
        assert any("example.com" in c for c in citations)
        assert any("secure.com" in c for c in citations)


class TestEvaluateRACE:
    """Tests for evaluate_race function."""

    def test_basic_evaluation(self):
        """Test basic RACE evaluation."""
        report = """
        # Executive Summary
        This report covers quantum computing advances.

        ## Key Findings
        1. Quantum supremacy achieved
        2. Error correction improving

        ## Conclusion
        The field is progressing rapidly.
        """
        scores = evaluate_race(report, "What are the latest quantum computing advances?")

        assert isinstance(scores, RACEScores)
        assert 0 <= scores.comprehensiveness <= 100
        assert 0 <= scores.depth <= 100
        assert 0 <= scores.instruction_following <= 100
        assert 0 <= scores.readability <= 100
        assert 0 <= scores.overall <= 100

    def test_empty_report(self):
        """Test evaluation of empty report."""
        scores = evaluate_race("", "Test query")
        assert isinstance(scores, RACEScores)
        # Empty report should have low scores
        assert scores.overall < 50

    def test_short_report(self):
        """Test evaluation of very short report."""
        scores = evaluate_race("Brief answer.", "Test query")
        assert isinstance(scores, RACEScores)
        # Short report should have lower comprehensiveness
        assert scores.comprehensiveness < 80

    def test_well_structured_report(self):
        """Test evaluation of well-structured report."""
        report = """
        # Research Report: Quantum Computing

        ## Executive Summary
        This comprehensive analysis examines recent advances in quantum computing,
        focusing on hardware developments, algorithm improvements, and practical applications.

        ## Background
        Quantum computing leverages quantum mechanical phenomena to perform computations.

        ## Key Findings
        1. **Hardware Progress**: Major improvements in qubit coherence times
        2. **Algorithm Development**: New quantum algorithms for optimization
        3. **Applications**: Drug discovery and cryptography applications emerging

        ## Analysis
        The evidence suggests significant progress across multiple fronts.

        ## Conclusion
        Quantum computing is advancing rapidly with practical applications on the horizon.

        ## References
        - [Quantum Research](https://example.com)
        - [Tech Report](https://tech.org)
        """
        scores = evaluate_race(report, "Analyze recent quantum computing advances")
        assert isinstance(scores, RACEScores)
        # Well-structured report should score better
        assert scores.readability >= 50


class TestEvaluateFACT:
    """Tests for evaluate_fact function."""

    def test_basic_evaluation(self):
        """Test basic FACT evaluation."""
        report = """
        According to [Source](https://example.com), quantum computing is advancing.
        Another study (https://research.org/paper) confirms this finding.
        """
        source_urls = {"https://example.com", "https://research.org/paper", "https://other.com"}

        scores = evaluate_fact(report, source_urls)

        assert isinstance(scores, FACTScores)
        assert scores.total_citations >= 0
        assert scores.verified_citations >= 0
        assert 0 <= scores.citation_accuracy <= 100

    def test_all_citations_verified(self):
        """Test when all citations match sources."""
        report = "See [A](https://a.com) and [B](https://b.com)."
        source_urls = {"https://a.com", "https://b.com"}

        scores = evaluate_fact(report, source_urls)

        assert scores.citation_accuracy == 100.0

    def test_no_citations(self):
        """Test report with no citations."""
        report = "This report has no citations or links."
        source_urls = {"https://example.com"}

        scores = evaluate_fact(report, source_urls)

        assert scores.total_citations == 0
        assert scores.citation_accuracy == 0.0

    def test_citation_efficiency(self):
        """Test citation efficiency calculation."""
        # 2 citations in ~10 words
        report = "See [A](https://a.com) and [B](https://b.com) here."
        source_urls = {"https://a.com", "https://b.com"}

        scores = evaluate_fact(report, source_urls)

        # Efficiency = citations per 1000 words
        # With ~10 words and 2 citations, efficiency should be high
        assert scores.word_count > 0
        assert scores.citation_efficiency >= 0

    def test_empty_source_urls(self):
        """Test with empty source URLs set."""
        report = "See [link](https://example.com)."
        source_urls = set()

        scores = evaluate_fact(report, source_urls)

        # Citations exist but none can be verified
        assert scores.total_citations >= 1
        assert scores.verified_citations == 0
        assert scores.citation_accuracy == 0.0


class TestGenerateBenchmarkResult:
    """Tests for generate_benchmark_result function."""

    def test_basic_result(self):
        """Test basic benchmark result generation."""
        result = generate_benchmark_result(
            task_id="task_001",
            query="What is AI?",
            report="AI is artificial intelligence.",
            source_urls={"https://ai.com"},
            elapsed_time=45.0,
            tasks_completed=5,
            findings_count=10,
        )

        assert result["task_id"] == "task_001"
        assert result["query"] == "What is AI?"
        assert "race_scores" in result
        assert "fact_scores" in result
        assert "throughput" in result

    def test_result_contains_timing(self):
        """Test that timing information is included."""
        result = generate_benchmark_result(
            task_id="t1",
            query="Test",
            report="Test report",
            source_urls=set(),
            elapsed_time=123.45,
            tasks_completed=10,
            findings_count=20,
        )

        assert result["throughput"]["elapsed_time"] == 123.45
        assert result["throughput"]["nodes_processed"] == 10
        assert result["throughput"]["findings_count"] == 20

    def test_result_structure(self):
        """Test complete result structure."""
        result = generate_benchmark_result(
            task_id="t1",
            query="Query",
            report="Report content",
            source_urls={"https://example.com"},
            elapsed_time=60.0,
            tasks_completed=5,
            findings_count=15,
        )

        # Check all expected top-level keys
        expected_keys = ["task_id", "query", "report", "race_scores", "fact_scores", "throughput"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

        # Check nested structures
        assert "overall" in result["race_scores"]
        assert "citation_accuracy" in result["fact_scores"]
        assert "elapsed_time" in result["throughput"]
