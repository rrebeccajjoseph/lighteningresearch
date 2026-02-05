"""Pytest configuration and shared fixtures."""

import os
import pytest
import tempfile
from unittest.mock import patch


@pytest.fixture(autouse=True)
def mock_api_keys():
    """
    Automatically mock API keys for all tests.

    This prevents tests from requiring real API keys and
    accidentally making real API calls.
    """
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-openai-key-for-testing",
        "TAVILY_API_KEY": "test-tavily-key-for-testing",
        "LANGCHAIN_TRACING_V2": "false",
    }):
        yield


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_report():
    """Provide a sample research report for testing."""
    return """
    # Research Report: Quantum Computing Advances

    ## Executive Summary
    This report analyzes recent developments in quantum computing technology,
    focusing on hardware improvements and algorithm development.

    ## Key Findings

    1. **Hardware Progress**
       According to [IBM Research](https://ibm.com/quantum), quantum computers
       have achieved significant improvements in qubit coherence times.

    2. **Algorithm Development**
       New quantum algorithms have been developed for optimization problems.
       Source: https://arxiv.org/quantum-algorithms

    3. **Practical Applications**
       Drug discovery and cryptography remain the most promising applications.

    ## Analysis
    The evidence suggests quantum computing is progressing rapidly, with
    practical applications expected within the next decade.

    ## Conclusion
    Quantum computing represents a transformational technology with
    significant investment from major technology companies.

    ## References
    - [IBM Quantum](https://ibm.com/quantum)
    - [Arxiv](https://arxiv.org/quantum-algorithms)
    - [Google AI](https://ai.google/quantum)
    """


@pytest.fixture
def sample_findings():
    """Provide sample findings for testing."""
    return [
        {
            "url": "https://ibm.com/quantum",
            "title": "IBM Quantum Computing",
            "content": "IBM has made significant advances in quantum computing...",
            "score": 0.9,
        },
        {
            "url": "https://arxiv.org/quantum-algorithms",
            "title": "Quantum Algorithms Survey",
            "content": "This paper surveys recent quantum algorithm developments...",
            "score": 0.85,
        },
        {
            "url": "https://ai.google/quantum",
            "title": "Google Quantum AI",
            "content": "Google achieved quantum supremacy in 2019...",
            "score": 0.8,
        },
    ]


@pytest.fixture
def sample_source_urls():
    """Provide sample source URLs for testing."""
    return {
        "https://ibm.com/quantum",
        "https://arxiv.org/quantum-algorithms",
        "https://ai.google/quantum",
        "https://nature.com/quantum",
        "https://science.org/quantum",
    }


@pytest.fixture
def sample_questions():
    """Provide sample research questions for testing."""
    from lighteningresearch.experiments.datasets import ResearchyQuestion

    return [
        ResearchyQuestion(
            id="test_001",
            query="What are the latest advances in quantum computing?",
            field="Physics",
            difficulty="medium",
        ),
        ResearchyQuestion(
            id="test_002",
            query="How does CRISPR gene editing work?",
            field="Biology",
            difficulty="hard",
        ),
    ]


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response factory."""
    class MockResponse:
        def __init__(self, content):
            self.content = content

    def factory(content):
        return MockResponse(content)

    return factory
