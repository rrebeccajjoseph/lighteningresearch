from langgraph.graph import StateGraph, END
from .models import FRState
from .nodes import planner, dispatch, worker, orchestrator, synthesize

def should_continue(state: FRState) -> str:
    if state["stop"]:
        return "synthesize"
    if state["pending"] or state["in_flight"] > 0:
        return "dispatch"
    return "synthesize"

def build_app():
    g = StateGraph(FRState)

    g.add_node("planner", planner)
    g.add_node("dispatch", dispatch)
    g.add_node("worker", worker)
    g.add_node("orchestrator", orchestrator)
    g.add_node("synthesize", synthesize)

    g.set_entry_point("planner")
    g.add_edge("planner", "dispatch")

    # dispatch uses Command to route to workers or orchestrator
    # worker uses Command to route back to orchestrator
    # No explicit edges needed - Command handles routing

    g.add_conditional_edges(
        "orchestrator",
        should_continue,
        {"dispatch": "dispatch", "synthesize": "synthesize"}
    )

    g.add_edge("synthesize", END)
    return g.compile()