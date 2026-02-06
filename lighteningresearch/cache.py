"""
Reproducibility utilities for benchmarking.

Provides:
- Search result caching for reproducible benchmarks
- Static corpus support (like FineWeb)
- Result serialization/deserialization
"""

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from .utils.search_normalization import normalize_search_results


@dataclass
class CachedSearchResult:
    """A cached search result for reproducibility."""
    query: str
    query_hash: str
    timestamp: str
    results: List[Dict[str, Any]]


class SearchCache:
    """
    Cache search results for reproducible benchmarks.

    Usage:
        cache = SearchCache("./cache")

        # Check cache first
        results = cache.get(query)
        if results is None:
            results = await search_tool.ainvoke(query)
            cache.set(query, results)
    """

    def __init__(self, cache_dir: str = "./search_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.json"
        self._load_index()

    def _load_index(self):
        """Load cache index."""
        if self.index_file.exists():
            with open(self.index_file) as f:
                self.index = json.load(f)
        else:
            self.index = {"queries": {}, "created": datetime.now().isoformat()}

    def _save_index(self):
        """Save cache index."""
        with open(self.index_file, "w") as f:
            json.dump(self.index, f, indent=2)

    def _hash_query(self, query: str) -> str:
        """Create consistent hash for a query."""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]

    def get(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached results for a query."""
        query_hash = self._hash_query(query)
        cache_file = self.cache_dir / f"{query_hash}.json"

        if cache_file.exists():
            with open(cache_file) as f:
                cached = json.load(f)
                return cached.get("results", [])
        return None

    def set(self, query: str, results: List[Dict[str, Any]]):
        """Cache results for a query."""
        query_hash = self._hash_query(query)
        cache_file = self.cache_dir / f"{query_hash}.json"

        normalized = normalize_search_results(results)

        cached = CachedSearchResult(
            query=query,
            query_hash=query_hash,
            timestamp=datetime.now().isoformat(),
            results=normalized,
        )

        with open(cache_file, "w") as f:
            json.dump(asdict(cached), f, indent=2)

        self.index["queries"][query_hash] = {
            "query": query[:100],
            "timestamp": cached.timestamp,
            "result_count": len(normalized),
        }
        self._save_index()

    def has(self, query: str) -> bool:
        """Check if query is cached."""
        return self.get(query) is not None

    def clear(self):
        """Clear all cached results."""
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
        self.index = {"queries": {}, "created": datetime.now().isoformat()}
        self._save_index()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_queries": len(self.index.get("queries", {})),
            "cache_dir": str(self.cache_dir),
            "created": self.index.get("created"),
        }


class CachedSearchTool:
    """
    Wrapper that caches search tool results.

    Usage:
        from lighteningresearch.tools import make_search_tool
        from lighteningresearch.cache import CachedSearchTool

        base_tool = make_search_tool()
        cached_tool = CachedSearchTool(base_tool, cache_dir="./cache")

        # Will use cache if available, otherwise calls base tool
        results = await cached_tool.ainvoke(query)
    """

    def __init__(self, base_tool, cache_dir: str = "./search_cache"):
        self.base_tool = base_tool
        self.cache = SearchCache(cache_dir)

    async def ainvoke(self, query: str) -> List[Dict[str, Any]]:
        """Search with caching."""
        cached = self.cache.get(query)
        if cached is not None:
            return cached

        results = await self.base_tool.ainvoke(query)
        self.cache.set(query, results)

        # Return normalized format
        return normalize_search_results(results)

    def invoke(self, query: str) -> List[Dict[str, Any]]:
        """Sync search with caching."""
        import asyncio
        return asyncio.run(self.ainvoke(query))


def create_reproducible_corpus(
    queries: List[str],
    output_dir: str,
    search_tool,
) -> Dict[str, Any]:
    """
    Pre-fetch and cache search results for a set of queries.

    This creates a static corpus for reproducible benchmarking,
    similar to FineWeb in the DeepResearch Bench paper.

    Usage:
        queries = ["quantum computing", "CRISPR", "climate change"]
        corpus = create_reproducible_corpus(queries, "./corpus", search_tool)
    """
    import asyncio

    cache = SearchCache(output_dir)
    stats = {"total_queries": 0, "total_results": 0, "queries": []}

    async def fetch_all():
        for query in queries:
            if not cache.has(query):
                print(f"Fetching: {query[:50]}...")
                try:
                    results = await search_tool.ainvoke(query)
                    cache.set(query, results)

                    if isinstance(results, dict):
                        results = results.get("results") or []

                    stats["total_results"] += len(results)
                except Exception as e:
                    print(f"  Error: {e}")

            stats["total_queries"] += 1
            stats["queries"].append(query)

    try:
        asyncio.get_running_loop()
        has_loop = True
    except RuntimeError:
        has_loop = False

    if not has_loop:
        asyncio.run(fetch_all())
    else:
        result: Dict[str, Any] = {}
        error: Optional[BaseException] = None

        def runner():
            nonlocal result, error
            try:
                asyncio.run(fetch_all())
                result = {}
            except BaseException as exc:
                error = exc

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()
        if error:
            raise error

    # Save corpus metadata
    metadata = {
        "created": datetime.now().isoformat(),
        "total_queries": stats["total_queries"],
        "total_results": stats["total_results"],
        "queries": stats["queries"],
    }

    with open(f"{output_dir}/corpus_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nCorpus created: {stats['total_queries']} queries, {stats['total_results']} results")
    return metadata
