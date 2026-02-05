"""Tests for lighteningresearch.models module."""

import pytest
from typing import Set

from lighteningresearch.models import (
    FRState,
    Task,
    Finding,
    union_sets,
)


class TestUnionSets:
    """Tests for union_sets reducer function."""

    def test_union_two_sets(self):
        """Test merging two sets."""
        a = {"url1", "url2"}
        b = {"url2", "url3"}
        result = union_sets(a, b)
        assert result == {"url1", "url2", "url3"}

    def test_union_empty_sets(self):
        """Test merging empty sets."""
        result = union_sets(set(), set())
        assert result == set()

    def test_union_with_empty(self):
        """Test merging with one empty set."""
        a = {"url1", "url2"}
        result = union_sets(a, set())
        assert result == {"url1", "url2"}

        result = union_sets(set(), a)
        assert result == {"url1", "url2"}

    def test_union_identical_sets(self):
        """Test merging identical sets."""
        a = {"url1", "url2"}
        result = union_sets(a, a.copy())
        assert result == {"url1", "url2"}

    def test_union_preserves_types(self):
        """Test that union preserves string types."""
        a = {"http://example.com"}
        b = {"https://test.org"}
        result = union_sets(a, b)
        assert all(isinstance(url, str) for url in result)


class TestTask:
    """Tests for Task dataclass."""

    def test_task_creation(self):
        """Test basic task creation."""
        task = Task(
            query="What is quantum computing?",
            depth=1,
            parent_id="root",
        )
        assert task.query == "What is quantum computing?"
        assert task.depth == 1
        assert task.parent_id == "root"

    def test_task_equality(self):
        """Test task equality comparison."""
        task1 = Task(query="test", depth=1, parent_id="p1")
        task2 = Task(query="test", depth=1, parent_id="p1")
        task3 = Task(query="different", depth=1, parent_id="p1")

        assert task1 == task2
        assert task1 != task3

    def test_task_depth_values(self):
        """Test task depth can be various values."""
        for depth in [0, 1, 2, 5, 10]:
            task = Task(query="test", depth=depth, parent_id="p")
            assert task.depth == depth


class TestFinding:
    """Tests for Finding dataclass."""

    def test_finding_creation(self):
        """Test basic finding creation."""
        finding = Finding(
            url="https://example.com/article",
            title="Example Article",
            content="This is the article content.",
            score=0.85,
            task_id="task_001",
        )
        assert finding.url == "https://example.com/article"
        assert finding.title == "Example Article"
        assert finding.content == "This is the article content."
        assert finding.score == 0.85
        assert finding.task_id == "task_001"

    def test_finding_score_bounds(self):
        """Test finding score can be various values."""
        # Score should typically be 0-1 but the dataclass doesn't enforce
        finding_low = Finding(
            url="u", title="t", content="c", score=0.0, task_id="t1"
        )
        finding_high = Finding(
            url="u", title="t", content="c", score=1.0, task_id="t1"
        )
        finding_mid = Finding(
            url="u", title="t", content="c", score=0.5, task_id="t1"
        )

        assert finding_low.score == 0.0
        assert finding_high.score == 1.0
        assert finding_mid.score == 0.5

    def test_finding_empty_content(self):
        """Test finding with empty content."""
        finding = Finding(
            url="https://example.com",
            title="Title",
            content="",
            score=0.0,
            task_id="t1",
        )
        assert finding.content == ""

    def test_finding_with_special_characters(self):
        """Test finding handles special characters."""
        finding = Finding(
            url="https://example.com/path?q=test&foo=bar",
            title="Title with 'quotes' and \"double quotes\"",
            content="Content with\nnewlines\tand\ttabs",
            score=0.5,
            task_id="t1",
        )
        assert "?" in finding.url
        assert "'" in finding.title
        assert "\n" in finding.content


class TestFRState:
    """Tests for FRState TypedDict structure."""

    def test_state_structure(self):
        """Test that FRState has expected keys."""
        # FRState is a TypedDict, we verify its structure
        required_keys = [
            "messages",
            "root_query",
            "config",
            "start_time",
            "time_budget_s",
            "stop",
            "max_depth",
            "max_breadth",
            "max_concurrency",
            "pending",
            "in_flight",
            "child_tasks",
            "findings",
            "seen_urls",
            "best_score",
            "stop_threshold",
            "tasks_dispatched",
            "tasks_completed",
            "final_report",
        ]

        # Get annotations from FRState
        annotations = FRState.__annotations__
        for key in required_keys:
            assert key in annotations, f"Missing key: {key}"

    def test_minimal_state_creation(self):
        """Test creating a minimal valid state dict."""
        state: FRState = {
            "messages": [],
            "root_query": "test query",
            "config": None,
            "start_time": 0.0,
            "time_budget_s": 60,
            "stop": False,
            "max_depth": 2,
            "max_breadth": 5,
            "max_concurrency": 4,
            "pending": [],
            "in_flight": 0,
            "child_tasks": [],
            "findings": [],
            "seen_urls": set(),
            "best_score": 0.0,
            "stop_threshold": 0.85,
            "tasks_dispatched": 0,
            "tasks_completed": 0,
            "final_report": "",
        }

        assert state["root_query"] == "test query"
        assert state["time_budget_s"] == 60
        assert isinstance(state["seen_urls"], set)
        assert isinstance(state["findings"], list)
