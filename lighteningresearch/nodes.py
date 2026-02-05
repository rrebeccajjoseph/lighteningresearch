"""
Node implementations for LightningResearch agent.

Each node is a function that takes state and returns state updates.
Configuration is read from state["config"] (AgentConfig instance).
"""

import asyncio
import time
import random
import uuid
from typing import Any, Dict, List, Set, Optional
from langgraph.types import Send, Command
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .models import Task, Finding, FRState
from .memory import update_memory_async
from .tools import make_search_tool
from .cache import CachedSearchTool
from .config import AgentConfig, ModelConfig, PromptConfig, ReportConfig


# =============================================================================
# Helpers
# =============================================================================

def get_config(state: Dict[str, Any]) -> AgentConfig:
    """Get AgentConfig from state, or return defaults."""
    return state.get("config") or AgentConfig()


def get_llm(model: str, temperature: float) -> ChatOpenAI:
    """Create an LLM instance with given settings."""
    return ChatOpenAI(model=model, temperature=temperature)


def time_left(state: Dict[str, Any]) -> float:
    """Calculate remaining time budget."""
    return state["time_budget_s"] - (time.time() - state["start_time"])


async def retry_search(search_tool, query: str, config: AgentConfig):
    """Retry search with exponential backoff."""
    attempts = max(1, config.retry_max_attempts)
    backoff = max(0.0, config.retry_initial_backoff_s)
    max_backoff = max(backoff, config.retry_max_backoff_s)

    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return await search_tool.ainvoke(query)
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            # Jittered exponential backoff
            jitter = random.uniform(0.0, 0.25 * backoff) if backoff > 0 else 0.0
            await asyncio.sleep(backoff + jitter)
            backoff = min(max_backoff, backoff * 2 if backoff > 0 else 1.0)

    raise last_exc if last_exc else RuntimeError("Search failed without exception")


def new_task(query: str, depth: int, parent_id: Optional[str] = None) -> Task:
    """Create a new research task."""
    return Task(id=str(uuid.uuid4()), query=query, depth=depth, parent_id=parent_id)


# =============================================================================
# Planner Node
# =============================================================================

def planner(state: FRState) -> Dict[str, Any]:
    """
    Generate diverse subqueries from the root query.

    Uses configurable:
    - Model (config.models.planner_model)
    - System prompt (config.prompts.planner_system)
    - Template (config.prompts.planner_template)
    """
    config = get_config(state)
    q = state["root_query"]

    # Adaptive breadth/depth based on query complexity
    broad = len(q.split()) > 6
    max_breadth = state["max_breadth"] if broad else max(2, state["max_breadth"] // 2)
    max_depth = state["max_depth"] if not broad else max(1, state["max_depth"] - 1)

    # Use configured model and prompts
    llm = get_llm(config.models.planner_model, config.models.planner_temperature)

    prompt = config.prompts.planner_template.format(
        max_breadth=max_breadth,
        query=q
    )

    resp = llm.invoke([
        SystemMessage(content=config.prompts.planner_system),
        HumanMessage(content=prompt)
    ])

    subqs = [line.strip() for line in resp.content.split("\n") if line.strip()][:max_breadth]

    return {
        "max_breadth": max_breadth,
        "max_depth": max_depth,
        "pending": [new_task(sq, 0) for sq in subqs],
    }


# =============================================================================
# Dispatch Node
# =============================================================================

def dispatch(state: FRState) -> Command:
    """
    Distribute pending tasks to workers.

    Uses LangGraph's Command/Send for dynamic parallel execution.
    """
    config = get_config(state)

    if state["stop"] or time_left(state) <= 0:
        return Command(goto="orchestrator", update={})

    tasks = state["pending"]
    if not tasks:
        return Command(goto="orchestrator", update={})

    # Pass required context to each worker
    worker_context = {
        "root_query": state["root_query"],
        "start_time": state["start_time"],
        "time_budget_s": state["time_budget_s"],
        "max_depth": state["max_depth"],
        "seen_urls": state["seen_urls"],
        "stop": state["stop"],
        "config": config,  # Pass config to workers
    }
    sends = [Send("worker", {"task": t, **worker_context}) for t in tasks]

    return Command(
        goto=sends,
        update={
            "pending": [],
            "in_flight": len(sends),
            "tasks_dispatched": len(sends),
        }
    )


# =============================================================================
# Worker Node
# =============================================================================

async def worker(input_state: Dict[str, Any]) -> Command:
    """
    Execute web search and score results.

    Uses configurable:
    - Search settings (config.search)
    - Scorer model (config.models.scorer_model)
    - Child task threshold (config.child_task_threshold)
    """
    config = get_config(input_state)
    task: Task = input_state["task"]

    if input_state.get("stop") or time_left(input_state) <= 0:
        return Command(goto="orchestrator", update={"in_flight": -1})

    # Create search tool with configured settings
    base_tool = make_search_tool(
        max_results=config.search.max_results_per_search,
        include_domains=config.search.include_domains,
        exclude_domains=config.search.exclude_domains,
    )
    search_tool = (
        CachedSearchTool(base_tool, cache_dir=config.search.cache_dir)
        if config.search.cache_enabled
        else base_tool
    )

    try:
        results = await retry_search(search_tool, task.query, config)
    except Exception:
        return Command(goto="orchestrator", update={"in_flight": -1})

    new_findings: List[Finding] = []
    new_child_tasks: List[Task] = []
    new_urls: Set[str] = set()

    seen_urls = input_state.get("seen_urls", set())
    root_query = input_state["root_query"]
    max_depth = input_state["max_depth"]

    # Normalize Tavily result shapes
    if isinstance(results, dict):
        results_iter = results.get("results") or results.get("data") or []
    else:
        results_iter = results

    # Create scorer LLM
    scorer_llm = get_llm(config.models.scorer_model, config.models.scorer_temperature)

    for r in (results_iter or []):
        if isinstance(r, str):
            url = r if r.startswith("http") else ""
            title = ""
            content = ""
        elif isinstance(r, dict):
            url = r.get("url") or ""
            title = r.get("title") or ""
            content = (r.get("content") or r.get("snippet") or "").strip()
        else:
            continue

        if not url or url in seen_urls or url in new_urls:
            continue
        if not content:
            continue

        # Score using configured prompt
        score_prompt = config.prompts.scorer_template.format(
            root_query=root_query,
            title=title,
            content=content[:1200]
        )
        try:
            score_resp = scorer_llm.invoke([
                SystemMessage(content=config.prompts.scorer_system),
                HumanMessage(content=score_prompt)
            ])
            s = float(str(score_resp.content).strip())
            s = max(0.0, min(1.0, s))
        except Exception:
            s = 0.3

        new_urls.add(url)

        new_findings.append(Finding(
            task_id=task.id,
            query=task.query,
            url=url,
            title=title,
            content=content,
            score=s
        ))

        # Spawn child task for high-quality results
        if s >= config.child_task_threshold and task.depth + 1 <= max_depth:
            child_query = config.prompts.child_task_template.format(
                topic=title or url,
                focus="key evidence and supporting details",
                root_query=root_query
            ) if hasattr(config.prompts, 'child_task_template') else \
                f"Find more evidence from {url} relevant to: {root_query}"

            new_child_tasks.append(new_task(
                query=child_query,
                depth=task.depth + 1,
                parent_id=task.id
            ))

    if config.memory.enabled and new_findings:
        await update_memory_async(
            path=config.memory.path,
            root_query=root_query,
            task_query=task.query,
            findings=new_findings,
            min_score=config.memory.min_score,
            max_sources=config.memory.max_sources,
        )

    return Command(
        goto="orchestrator",
        update={
            "findings": new_findings,
            "seen_urls": new_urls,
            "child_tasks": new_child_tasks,
            "in_flight": -1,
            "tasks_completed": 1,
        }
    )


# =============================================================================
# Orchestrator Node
# =============================================================================

def orchestrator(state: FRState) -> Dict[str, Any]:
    """
    Track quality and manage research loop.

    Stops when:
    - Best finding score >= stop_threshold
    - Time budget exhausted
    - No more pending tasks
    """
    best = state["best_score"]
    for f in state["findings"]:
        if f.score > best:
            best = f.score

    child_tasks = state.get("child_tasks", [])

    if best >= state["stop_threshold"]:
        return {"best_score": best, "stop": True, "pending": [], "child_tasks": []}

    if time_left(state) <= 0:
        return {"best_score": best, "stop": True, "pending": [], "child_tasks": []}

    return {
        "best_score": best,
        "pending": list(child_tasks),
        "child_tasks": [],
    }


# =============================================================================
# Synthesize Node
# =============================================================================

def synthesize(state: FRState) -> Dict[str, Any]:
    """
    Generate structured research report from findings.

    Uses configurable:
    - Model (config.models.synthesizer_model)
    - Report structure (config.report)
    - System prompt (config.prompts.synthesizer_system)
    """
    config = get_config(state)

    # Get top findings
    top_n = config.top_findings_for_report
    top = sorted(state["findings"], key=lambda f: f.score, reverse=True)[:top_n]

    evidence = "\n".join(
        f"- {f.title} ({f.url})\n  {f.content[:500]}"
        for f in top
    )

    # Build report structure from config
    report_structure = config.report.to_prompt()

    # Build synthesis prompt
    prompt = config.prompts.synthesizer_template.format(
        query=state["root_query"],
        report_structure=report_structure,
        evidence=evidence
    )

    llm = get_llm(config.models.synthesizer_model, config.models.synthesizer_temperature)

    resp = llm.invoke([
        SystemMessage(content=config.prompts.synthesizer_system),
        HumanMessage(content=prompt)
    ])

    report_content = resp.content

    return {
        "final_report": report_content,
        "messages": [AIMessage(content=report_content)],
    }
