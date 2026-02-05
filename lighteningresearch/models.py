from __future__ import annotations
import operator
from dataclasses import dataclass
from typing import Optional, List, Set, TypedDict, Annotated, Any

from langgraph.graph.message import add_messages


def union_sets(a: Set[str], b: Set[str]) -> Set[str]:
    """Reducer to union sets."""
    return a.union(b)


@dataclass(frozen=True)
class Task:
    id: str
    query: str
    depth: int
    parent_id: Optional[str] = None

@dataclass(frozen=True)
class Finding:
    task_id: str
    query: str
    url: str
    title: str
    content: str
    score: float

class FRState(TypedDict):
    # messages field for LangGraph compliance
    messages: Annotated[list, add_messages]

    root_query: str

    # configuration (AgentConfig instance)
    config: Any  # AgentConfig - using Any to avoid circular import

    # control
    start_time: float
    time_budget_s: float
    stop: bool

    # planner outputs
    max_depth: int
    max_breadth: int
    max_concurrency: int

    # task pool
    pending: List[Task]  # dispatch clears, orchestrator refills from child_tasks
    in_flight: Annotated[int, operator.add]  # reducer: workers return -1 to decrement
    child_tasks: Annotated[List[Task], operator.add]  # reducer: workers add child tasks

    # results - use reducers for concurrent updates
    findings: Annotated[List[Finding], operator.add]
    seen_urls: Annotated[Set[str], union_sets]

    # orchestration
    best_score: float
    stop_threshold: float

    # throughput metrics
    tasks_dispatched: Annotated[int, operator.add]  # total tasks sent to workers
    tasks_completed: Annotated[int, operator.add]   # tasks that returned results

    # output
    final_report: str