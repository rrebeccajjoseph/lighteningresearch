"""
Normalize raw search-provider responses into a consistent shape.

Tavily (and other providers) may return results as:
- A ``dict`` with a ``"results"`` or ``"data"`` key wrapping a list
- A bare ``list`` of dicts (each with url / title / content|snippet)
- A bare ``list`` of URL strings

``normalize_search_results`` flattens all of these into a single
``list[dict]`` where every item has *url*, *title*, and *content* keys.
"""

from typing import Any, Dict, List


def normalize_search_results(results: Any) -> List[Dict[str, str]]:
    """
    Normalize search results into a flat list of dicts.

    Each returned dict has exactly three keys::

        {"url": str, "title": str, "content": str}

    Args:
        results: Raw return value from a search provider.  May be a
            ``dict`` wrapping a list, a ``list`` of dicts, or a ``list``
            of URL strings.

    Returns:
        A (possibly empty) list of normalised result dicts.
    """
    # Top-level unwrap: extract the inner list from wrapper dicts
    if isinstance(results, dict):
        items = results.get("results") or results.get("data") or []
    else:
        items = results

    normalized: List[Dict[str, str]] = []
    for r in items or []:
        if isinstance(r, str):
            normalized.append({
                "url": r if r.startswith("http") else "",
                "title": "",
                "content": "",
            })
        elif isinstance(r, dict):
            normalized.append({
                "url": r.get("url") or "",
                "title": r.get("title") or "",
                "content": (r.get("content") or r.get("snippet") or "").strip(),
            })
    return normalized
