"""Tests for lighteningresearch.cache module."""

import json
import os
import pytest
import tempfile
from pathlib import Path

from lighteningresearch.cache import (
    SearchCache,
    CachedSearchResult,
    CachedSearchTool,
    create_reproducible_corpus,
)


class TestCachedSearchResult:
    """Tests for CachedSearchResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = CachedSearchResult(
            query="test query",
            query_hash="abc123",
            timestamp="2024-01-01T00:00:00",
            results=[{"url": "https://example.com", "title": "Test", "content": "Content"}],
        )
        assert result.query == "test query"
        assert result.query_hash == "abc123"
        assert len(result.results) == 1


class TestSearchCache:
    """Tests for SearchCache class."""

    @pytest.fixture
    def cache_dir(self):
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def cache(self, cache_dir):
        """Create a SearchCache instance."""
        return SearchCache(cache_dir)

    def test_init_creates_directory(self, cache_dir):
        """Test that initialization creates the cache directory."""
        cache_path = os.path.join(cache_dir, "new_cache")
        cache = SearchCache(cache_path)
        assert os.path.exists(cache_path)

    def test_hash_query_consistent(self, cache):
        """Test that query hashing is consistent."""
        hash1 = cache._hash_query("test query")
        hash2 = cache._hash_query("test query")
        assert hash1 == hash2

    def test_hash_query_case_insensitive(self, cache):
        """Test that hashing is case-insensitive."""
        hash1 = cache._hash_query("Test Query")
        hash2 = cache._hash_query("test query")
        assert hash1 == hash2

    def test_hash_query_strips_whitespace(self, cache):
        """Test that hashing strips whitespace."""
        hash1 = cache._hash_query("  test query  ")
        hash2 = cache._hash_query("test query")
        assert hash1 == hash2

    def test_set_and_get(self, cache):
        """Test setting and getting cached results."""
        query = "quantum computing"
        results = [
            {"url": "https://example.com/1", "title": "Article 1", "content": "Content 1"},
            {"url": "https://example.com/2", "title": "Article 2", "content": "Content 2"},
        ]

        cache.set(query, results)
        retrieved = cache.get(query)

        assert retrieved is not None
        assert len(retrieved) == 2
        assert retrieved[0]["url"] == "https://example.com/1"

    def test_get_nonexistent(self, cache):
        """Test getting a query that doesn't exist."""
        result = cache.get("nonexistent query")
        assert result is None

    def test_has_existing(self, cache):
        """Test has() for existing query."""
        cache.set("test", [{"url": "u", "title": "t", "content": "c"}])
        assert cache.has("test") is True

    def test_has_nonexistent(self, cache):
        """Test has() for nonexistent query."""
        assert cache.has("nonexistent") is False

    def test_clear(self, cache):
        """Test clearing the cache."""
        cache.set("test1", [{"url": "u", "title": "t", "content": "c"}])
        cache.set("test2", [{"url": "u", "title": "t", "content": "c"}])

        assert cache.has("test1")
        assert cache.has("test2")

        cache.clear()

        assert not cache.has("test1")
        assert not cache.has("test2")

    def test_stats(self, cache):
        """Test getting cache statistics."""
        cache.set("test1", [{"url": "u", "title": "t", "content": "c"}])
        cache.set("test2", [{"url": "u", "title": "t", "content": "c"}])

        stats = cache.stats()

        assert stats["total_queries"] == 2
        assert "cache_dir" in stats
        assert "created" in stats

    def test_normalizes_dict_results(self, cache):
        """Test that dict-style results are normalized."""
        query = "test"
        # Tavily-style response with nested results
        results = {
            "results": [
                {"url": "https://example.com", "title": "Test", "content": "Content"}
            ]
        }

        cache.set(query, results)
        retrieved = cache.get(query)

        assert retrieved is not None
        assert len(retrieved) == 1
        assert retrieved[0]["url"] == "https://example.com"

    def test_handles_snippet_key(self, cache):
        """Test that 'snippet' is normalized to 'content'."""
        query = "test"
        results = [
            {"url": "https://example.com", "title": "Test", "snippet": "Snippet content"}
        ]

        cache.set(query, results)
        retrieved = cache.get(query)

        assert retrieved is not None
        assert retrieved[0]["content"] == "Snippet content"

    def test_index_persistence(self, cache_dir):
        """Test that index persists across instances."""
        cache1 = SearchCache(cache_dir)
        cache1.set("test", [{"url": "u", "title": "t", "content": "c"}])

        # Create new instance pointing to same directory
        cache2 = SearchCache(cache_dir)

        assert cache2.has("test")

    def test_empty_results(self, cache):
        """Test caching empty results."""
        cache.set("empty query", [])
        retrieved = cache.get("empty query")

        assert retrieved is not None
        assert len(retrieved) == 0


class TestCachedSearchTool:
    """Tests for CachedSearchTool class."""

    @pytest.fixture
    def cache_dir(self):
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_init(self, cache_dir):
        """Test initialization."""
        class MockTool:
            async def ainvoke(self, query):
                return [{"url": "https://example.com", "title": "Test", "content": "Content"}]

        tool = CachedSearchTool(MockTool(), cache_dir)
        assert tool.base_tool is not None
        assert tool.cache is not None

    @pytest.mark.asyncio
    async def test_returns_cached_results(self, cache_dir):
        """Test that cached results are returned without calling base tool."""
        call_count = 0

        class MockTool:
            async def ainvoke(self, query):
                nonlocal call_count
                call_count += 1
                return [{"url": "https://example.com", "title": "Test", "content": "Content"}]

        tool = CachedSearchTool(MockTool(), cache_dir)

        # First call should hit the base tool
        result1 = await tool.ainvoke("test query")
        assert call_count == 1
        assert len(result1) == 1

        # Second call should use cache
        result2 = await tool.ainvoke("test query")
        assert call_count == 1  # Still 1, not called again
        assert len(result2) == 1

    @pytest.mark.asyncio
    async def test_caches_new_results(self, cache_dir):
        """Test that new results are cached."""
        class MockTool:
            async def ainvoke(self, query):
                return [{"url": "https://example.com", "title": "Test", "content": "Content"}]

        tool = CachedSearchTool(MockTool(), cache_dir)

        await tool.ainvoke("new query")

        # Check cache directly
        assert tool.cache.has("new query")


class TestCreateReproducibleCorpus:
    """Tests for create_reproducible_corpus function."""

    @pytest.fixture
    def output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.asyncio
    async def test_creates_corpus(self, output_dir):
        """Test creating a reproducible corpus."""
        class MockTool:
            async def ainvoke(self, query):
                return [
                    {"url": f"https://example.com/{query}", "title": f"Result for {query}", "content": "Content"}
                ]

        queries = ["query1", "query2"]
        metadata = create_reproducible_corpus(queries, output_dir, MockTool())

        assert metadata["total_queries"] == 2
        assert os.path.exists(os.path.join(output_dir, "corpus_metadata.json"))

    @pytest.mark.asyncio
    async def test_skips_cached_queries(self, output_dir):
        """Test that already-cached queries are skipped."""
        call_count = 0

        class MockTool:
            async def ainvoke(self, query):
                nonlocal call_count
                call_count += 1
                return [{"url": "https://example.com", "title": "Test", "content": "Content"}]

        # First run
        create_reproducible_corpus(["query1"], output_dir, MockTool())
        assert call_count == 1

        # Second run should skip cached query
        create_reproducible_corpus(["query1"], output_dir, MockTool())
        assert call_count == 1  # Still 1, not called again
