"""
Lightweight memory for reusing high-quality sources across runs.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .models import Finding

_MEMORY_LOCK = asyncio.Lock()


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().strip().split())


def _default_memory() -> Dict[str, Any]:
    now = datetime.now().isoformat()
    return {
        "created": now,
        "updated": now,
        "sources": {},
        "queries": {},
    }


def _load_memory(path: str) -> Dict[str, Any]:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return _default_memory()
    try:
        with open(path) as f:
            data = json.load(f)
            if "sources" not in data or "queries" not in data:
                return _default_memory()
            return data
    except Exception:
        return _default_memory()


def _save_memory(path: str, data: Dict[str, Any]) -> None:
    path = os.path.abspath(path)
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    data["updated"] = datetime.now().isoformat()
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _prune_sources(data: Dict[str, Any], max_sources: int) -> None:
    sources = data.get("sources", {})
    if max_sources <= 0 or len(sources) <= max_sources:
        return

    ranked = sorted(
        sources.items(),
        key=lambda kv: float(kv[1].get("score", 0.0)),
        reverse=True,
    )
    keep_urls = set(url for url, _ in ranked[:max_sources])
    data["sources"] = {url: src for url, src in sources.items() if url in keep_urls}

    # Clean query -> url mappings
    queries = data.get("queries", {})
    for q, urls in list(queries.items()):
        filtered = [u for u in urls if u in keep_urls]
        if filtered:
            queries[q] = filtered
        else:
            queries.pop(q, None)
    data["queries"] = queries


def update_memory(
    path: str,
    root_query: str,
    task_query: Optional[str],
    findings: Iterable[Finding],
    min_score: float,
    max_sources: int,
    max_content_chars: int = 1200,
) -> None:
    data = _load_memory(path)
    sources = data.get("sources", {})
    queries = data.get("queries", {})

    query_keys: Set[str] = set()
    if root_query:
        query_keys.add(_normalize_query(root_query))
    if task_query:
        query_keys.add(_normalize_query(task_query))

    for f in findings:
        if f.score < min_score:
            continue
        if not f.url:
            continue

        entry = sources.get(f.url, {})
        entry["url"] = f.url
        entry["title"] = f.title or entry.get("title", "")
        if f.content:
            entry["content"] = f.content[:max_content_chars]
        entry["score"] = max(float(entry.get("score", 0.0)), float(f.score))
        entry["last_seen"] = datetime.now().isoformat()

        entry_queries = set(entry.get("queries", []))
        entry_queries.update(query_keys)
        entry["queries"] = sorted(entry_queries)
        sources[f.url] = entry

        for q in query_keys:
            urls = set(queries.get(q, []))
            urls.add(f.url)
            queries[q] = sorted(urls)

    data["sources"] = sources
    data["queries"] = queries
    _prune_sources(data, max_sources=max_sources)
    _save_memory(path, data)


async def update_memory_async(
    path: str,
    root_query: str,
    task_query: Optional[str],
    findings: Iterable[Finding],
    min_score: float,
    max_sources: int,
) -> None:
    async with _MEMORY_LOCK:
        await asyncio.to_thread(
            update_memory,
            path,
            root_query,
            task_query,
            list(findings),
            min_score,
            max_sources,
        )


def load_seed_findings(
    path: str,
    root_query: str,
    max_seed: int,
) -> Tuple[List[Finding], Set[str], float]:
    data = _load_memory(path)
    queries = data.get("queries", {})
    sources = data.get("sources", {})

    key = _normalize_query(root_query)
    urls = queries.get(key, [])
    if not urls:
        return [], set(), 0.0

    candidates = []
    for url in urls:
        entry = sources.get(url)
        if not entry:
            continue
        candidates.append(entry)

    candidates.sort(key=lambda e: float(e.get("score", 0.0)), reverse=True)
    candidates = candidates[:max_seed]

    findings = [
        Finding(
            task_id="memory",
            query=root_query,
            url=e.get("url", ""),
            title=e.get("title", ""),
            content=e.get("content", ""),
            score=float(e.get("score", 0.0)),
        )
        for e in candidates
    ]
    urls_set = {f.url for f in findings if f.url}
    best_score = max((f.score for f in findings), default=0.0)
    return findings, urls_set, best_score


def apply_memory_seed(state: Dict[str, Any], config: Any, root_query: str) -> Dict[str, Any]:
    if not getattr(config, "memory", None) or not config.memory.enabled:
        return state

    findings, urls, best_score = load_seed_findings(
        path=config.memory.path,
        root_query=root_query,
        max_seed=config.memory.max_seed_findings,
    )

    if findings:
        state["findings"] = list(state.get("findings", [])) + findings
        state["seen_urls"] = set(state.get("seen_urls", set())).union(urls)
        if config.memory.allow_early_stop:
            state["best_score"] = max(float(state.get("best_score", 0.0)), best_score)

    return state
