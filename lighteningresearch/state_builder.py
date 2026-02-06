"""
Shared state initialization for the LightningResearch agent.

Every entry point (CLI, benchmark, experiments) needs the same initial
state dictionary.  This module provides a single canonical builder so
the schema is defined in exactly one place.
"""

import time

from langchain_core.messages import HumanMessage

from .config import AgentConfig
from .memory import apply_memory_seed


def build_initial_state(query: str, config: AgentConfig) -> dict:
    """
    Build the initial agent state dictionary.

    Args:
        query: The research query string.
        config: Fully-configured AgentConfig instance.

    Returns:
        A state dict ready to pass into ``app.ainvoke``.
        Memory seeding (``apply_memory_seed``) is already applied.
    """
    state = {
        "messages": [HumanMessage(content=query)],
        "root_query": query,
        "config": config,

        "start_time": time.time(),
        "time_budget_s": config.time_budget_s,
        "stop": False,

        "max_depth": config.max_depth,
        "max_breadth": config.max_breadth,
        "max_concurrency": config.max_concurrency,

        "pending": [],
        "in_flight": 0,
        "child_tasks": [],

        "findings": [],
        "seen_urls": set(),

        "best_score": 0.0,
        "stop_threshold": config.stop_threshold,

        "tasks_dispatched": 0,
        "tasks_completed": 0,

        "final_report": "",
    }
    state = apply_memory_seed(state, config, query)
    return state
