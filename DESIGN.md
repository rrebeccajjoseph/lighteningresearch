# LighteningResearch

LightningResearch is a LangGraph based deep research agent that overcomes the high latency of sequential research systems by using adaptive planning and runtime orchestration to dynamically spawn, parallelize, and prune breadth/depth-wise web search branches from a user query.

## What did you try, what worked well, and what didn’t work well. How did this influence changes you made. Specific examples are good!

What worked well:

1. Reducers in LangGraph state
Using reducers let multiple workers update shared state at the same time without conflicts. The only awkward part is clearing lists, since reducers are additive by design, so the orchestrator has to manually reset them.

2. Command-based routing
Switching to Command made the system way more flexible. The orchestrator can fan out to any number of workers and workers can route back dynamically. 

3. Passing context through Send
Workers only get the context they need instead of the full graph state. That kept things clean and predictable, especially with parallel execution. It does mean you have to be explicit about what you pass, but that’s a good constraint.

4. Orchestrator pattern
Having a central orchestrator simplified the whole system. Workers do the work, the orchestrator decides when to stop, spawn tasks, or continue. It kept responsibilities clear and made the system easier to extend.


What didn’t work well

1. Result normalization was messy
Tavily responses came back in inconsistent shapes, so normalization logic ended up scattered and fragile. This should really live in a dedicated parser/validator.

2. Reducers make clearing state awkward
Reducers are great for accumulation but not for resets. Manually overwriting fields works, but it’s not intuitive and needs clearer patterns.

3. Timeouts live outside the graph
Right now time limits are enforced externally. It works, but the graph itself should own that logic.

4. Dedup is too shallow
Only deduping by URL misses duplicate content from different sites. Content-based dedup would be a big improvement.

5. No streaming output
Everything happens and then you get a final report. Streaming intermediate results would make the system feel much more responsive.


It pushed the design toward reliability over cleverness.

I leaned more on the orchestrator, separated concerns more cleanly, and started prioritizing retries, logging, and validation. Reducers and Command routing stayed — those were clear wins. 


## What are known shortcomings that you didn’t have time to address. How would you fix them if you had more time?

Messy result parsing
The Tavily response normalization is defensive and scattered. It works, but it’s brittle and hard to maintain.

How I’d fix it:
Create a dedicated parser layer with a strict schema and validation. Workers should only ever see normalized, typed results.

URL-only deduplication
Right now dedup just checks URLs, which misses reposted or syndicated content.

How I’d fix it:
Add content hashing or embeddings-based similarity so near-duplicate articles collapse into one finding.

Reducers make resets awkward
Reducers are great for accumulation but not for clearing state. The orchestrator manually resets fields, which isn’t obvious unless you know the pattern.

How I’d fix it:
Split state into two types:
	•	reducer-backed accumulation fields
	•	ephemeral loop state that gets overwritten each iteration

Timeouts live outside the graph
Time budgets are enforced by wrappers around execution, not by the graph itself.

How I’d fix it:
Move time tracking into the orchestrator so it can stop scheduling new work and gracefully wind down active tasks.

No streaming or incremental output
Everything happens silently until the final report, which makes the system feel slower and less transparent.

How I’d fix it:
Stream findings and scores as they arrive and update a live report. The orchestrator already tracks this info — it just needs to emit it.

Limited task planning quality
Task generation is mostly heuristic and prompt-driven. It works, but it’s not adaptive.

How I’d fix it:
Add feedback loops:
	•	learn which queries produce high-scoring findings
	•	bias future branching toward those patterns
	•	prune consistently low-yield task types



What are future features you would add if you had more time?

1. Memory benchamrk - I'd want to be able to test how much memoeyr is important to have a self improving deep research agent. Right now the assumption is just that it does help.

2. Getting it up on Langsmith studio - this was just something I didn't get time to do!

3. Source diversity: Track and balance source types (news, academic, etc.), so then that way you have a more robust answer


### System Diagram
```
                    ┌─────────────────────────────────────────┐
                    │              User Query                  │
                    └─────────────────┬───────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LangGraph Runtime                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                           FRState                                    │   │
│  │  messages, root_query, findings, seen_urls, pending, child_tasks... │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│      ┌───────────────────────────────┼───────────────────────────────┐     │
│      │                               │                               │     │
│      ▼                               ▼                               ▼     │
│  ┌────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐   ┌─────┐ │
│  │Planner │───▶│ Dispatch │───▶│ Workers  │───▶│ Orchestrator │──▶│Synth│ │
│  └────────┘    └──────────┘    │ (async)  │    └──────┬───────┘   └─────┘ │
│                     ▲          └──────────┘           │                    │
│                     │                                 │                    │
│                     └─────────────────────────────────┘                    │
│                              (loop while pending)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │         Structured Research Report       │
                    │    + RACE Scores + FACT Metrics          │
                    └─────────────────────────────────────────┘
```


## State Schema
1. FRState - shared state object that flows through the graph. Each node (planner, dispatch, worker, orchestrator, syntehesize) reads from and wrties to this state

class FRState(TypedDict): 

    # LangGraph compliance - required for message handling
    messages: Annotated[list, add_messages]

    # Core - the original user query that drives the research
    root_query: str

    # Control
    start_time: float # when research started
    time_budget_s: float # how long to run research for
    stop: bool #early stopping flag

    # Research parameters
    max_depth: int # How deep to explore (e.g., 2 levels)
    max_breadth: int #max subqueries per level
    max_concurrency: int #parallel workers at once

    # Task management
    pending: List[Task] #tasks waiting to run
    in_flight: Annotated[int, operator.add] #currently running tasks
    child_tasks: Annotated[List[Task], operator.add] #new tasks spawned

    # Results
    findings: Annotated[List[Finding], operator.add] #research results found
    seen_urls: Annotated[Set[str], union_sets] #URLs already processed

    # Orchestration
    best_score: float # Highest quality score seen so far
    stop_threshold: float # Quality threshold to stop early (e.g., 0.85)

    # Throughput metrics
    tasks_dispatched: Annotated[int, operator.add] # Total tasks sent
    tasks_completed: Annotated[int, operator.add] # Tasks finished

    # Output
    final_report: str



