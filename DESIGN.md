# LighteningResearch

LightningResearch is a LangGraph based deep research agent that usese adaptive planning and runtime orchestration to dynamically spawn, parallelize, and prune breadth/depth-wise web search branches from a user query. 

## What did you try, what worked well, and what didn’t work well. How did this influence changes you made. Specific examples are good!

**Initial approach: Direct worker-to-worker communication**
At first, I had workers calling each other directly and sharing the full graph state. This got messy fast—workers were tightly coupled and it was hard to track what was happening when things ran in parallel.

**What I changed:** Introduced the orchestrator pattern with Command-based routing. Now workers report back to a central orchestrator that decides what happens next. This cleaned up responsibilities and made the flow way easier to follow.

**Early state management: Simple shared dictionaries**
I started with basic state updates where workers would just write to shared fields. This immediately broke when multiple workers tried updating the same thing simultaneously race conditions everywhere.

**What I changed:** Switched to reducers for anything that needed concurrent updates. Now workers can safely append results in parallel without conflicts. The only rough edge is clearing lists, since reducers are additive—I have to manually reset them in the orchestrator, which feels a bit hacky but works.


**Context passing: Full graph state everywhere**
Every worker got the entire graph state at first. This worked but felt wasteful and made it hard to reason about what each worker actually depended on. With parallel execution, it also made state mutations unpredictable.

**What I changed:** Started using Send to pass only the specific context each worker needs. More deliberate, but that's actually good—forces you to think about dependencies and keeps parallel execution predictable.


**Timeout management: External wrapper**
Initially had a simple timeout wrapper around the whole graph execution—just kill everything after N seconds. This worked but the graph had no awareness of time pressure, so it couldn't make smart decisions about prioritization.

**What I changed:** Moved timeout tracking into the graph state itself with a `time_left()` helper. Now the orchestrator and workers can check remaining time and decide whether to spawn more work or start wrapping up.


**What still needs work:**

**Result normalization is scattered**
Tavily responses come back in inconsistent shapes, and right now normalization logic lives in the worker code. Would be cleaner to have a dedicated parser that validates and normalizes before results hit the graph.

**Deduplication is too simple**
Currently just deduping by URL with `seen_urls`. This misses duplicate content from different sources—same article from multiple aggregators shows up multiple times. Content-based dedup would help but isn't implemented yet.

**No streaming feedback**
The graph runs and then prints a final report. Everything happens in a black box. Streaming intermediate results would make it feel more responsive and help with debugging or even human in the loop features.


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



