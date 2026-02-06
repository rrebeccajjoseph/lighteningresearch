"""
Centralized LLM instance creation for LightningResearch.

All ChatOpenAI instantiation is routed through this module so that
model defaults, constructor arguments, and the underlying provider
are defined in exactly one place.
"""

from langchain_openai import ChatOpenAI


def get_research_llm(
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> ChatOpenAI:
    """Create an LLM for research tasks (planning, synthesis, report generation)."""
    return ChatOpenAI(model=model, temperature=temperature)


def get_eval_llm(
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
) -> ChatOpenAI:
    """Create an LLM for evaluation tasks (RACE scoring, report judging)."""
    return ChatOpenAI(model=model, temperature=temperature)


def get_fast_llm(
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
) -> ChatOpenAI:
    """Create an LLM for fast, deterministic tasks (relevance scoring)."""
    return ChatOpenAI(model=model, temperature=temperature)
