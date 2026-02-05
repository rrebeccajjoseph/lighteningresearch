"""
Search tool wrappers for LightningResearch.
"""

from typing import Optional, List
from langchain_tavily import TavilySearch


def make_search_tool(
    max_results: int = 5,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    search_depth: str = "basic",
):
    """
    Create a configured Tavily search tool.

    Args:
        max_results: Maximum number of results per search
        include_domains: Only search these domains (e.g., ["arxiv.org"])
        exclude_domains: Exclude these domains (e.g., ["pinterest.com"])
        search_depth: "basic" or "advanced"

    Returns:
        Configured TavilySearch instance
    """
    kwargs = {
        "max_results": max_results,
    }

    # Add optional filters if provided
    if include_domains:
        kwargs["include_domains"] = include_domains
    if exclude_domains:
        kwargs["exclude_domains"] = exclude_domains

    return TavilySearch(**kwargs)
