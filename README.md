# LightningResearch

LightningResearch is a LangGraph-based deep research agent that plans, searches, and synthesizes grounded research reports using parallel web search and adaptive reasoning.

The goal of this project is to demonstrate how a research agent can move beyond simple sequential prompting, and now has parallel orchestration, adaptive planning, and evaluation for real-world LLM systems.

---

# What this system does

Given a research question, LightningResearch:

1. Breaks the query into sub-questions  
2. Runs parallel web searches  
3. Scores relevance and quality using LLMs  
4. Explores promising leads more deeply  
5. Synthesizes a structured research report  
6. Evaluates output quality automatically  

The system is built on*LangGraph, enabling dynamic routing, concurrency, and stateful orchestration rather than a static DAG.

---

# Core Design Ideas

## Parallel-first research

Most research agents operate sequentially:  
search → wait → analyze → repeat.

LightningResearch instead runs multiple research branches concurrently and expands only the ones that look promising.

This improves:
- latency  
- coverage  
- resource efficiency  

## Adaptive planning

The agent dynamically decides:
- how many subqueries to spawn  
- how deep to explore  
- when to stop early  

This allows simple queries to resolve quickly and complex queries to expand.

## LLM-based scoring

Search results and intermediate findings are scored for relevance and quality.  
Low-signal branches naturally terminate early.

## Configurable system

Nearly everything can be tuned:
- models per task (planner, scorer, synthesizer)  
- search depth/breadth  
- time budget  
- domains  
- report structure  
- prompts  

## Built-in evaluation

The system includes automated evaluation:

### RACE
- Comprehensiveness  
- Depth  
- Instruction following  
- Readability  

### FACT
- Citation accuracy  
- Citation efficiency  
- Source verification  

This makes it easier to treat research quality as something measurable.

---

# Quick Start

## 1. Install
```bash
pip install -e .

2. Configure environment

cp .env.example .env

Add keys:

OPENAI_API_KEY=
TAVILY_API_KEY=


3. Run

python run.py "What are the latest advances in quantum computing?"

⸻

Basic Usage

Time presets

# ~45 seconds
python run.py "Your question"

# ~2 minutes
python run.py --time standard "Your question"

# ~10 minutes
python run.py --time deep "Your question"

Custom budget:

python run.py --time-seconds 180 "Your question"


⸻

Configuration Presets

These tune search domains, depth, and report style.

Academic research

python run.py --preset academic "CRISPR applications in medicine"

Uses:
	•	deeper search
	•	academic domains
	•	structured report

News / current events

python run.py --preset news "Latest AI regulation developments"

Technical / developer research

python run.py --preset technical "React performance optimization"


⸻

Model Configuration

Use one model:

python run.py --model gpt-4o "Complex topic"

Or separate models:

python run.py \
  --planner-model gpt-4o \
  --synth-model gpt-4o-mini \
  "Your query"

This allows cost vs quality tradeoffs.

⸻

Search Controls

More results per search:

python run.py --max-results 10 "Your query"

Restrict domains:

python run.py \
  --include-domains arxiv.org nature.com \
  "Scientific topic"

Exclude domains:

python run.py \
  --exclude-domains pinterest.com facebook.com \
  "Your query"


⸻

Research Controls

Deeper exploration:

python run.py --max-depth 3 --max-breadth 8 "Complex topic"

Stop earlier:

python run.py --stop-threshold 0.7 "Simple topic"


⸻

Evaluation & Output

Run evaluation:

python run.py --eval "Your query"

Save results:

python run.py --eval --output results.json "Your query"

Quiet mode:

python run.py --quiet "Your query"


⸻

## Python Usage 
You don't just have to use

You can also use LightningResearch as a library.

from lighteningresearch import AgentConfig, ModelConfig, SearchConfig, build_app

config = AgentConfig(
    time_budget_s=300,
    max_depth=3,
    max_breadth=6,
    stop_threshold=0.8,

    models=ModelConfig(
        planner_model="gpt-4o",
        synthesizer_model="gpt-4o",
        scorer_model="gpt-4o-mini",
    ),

    search=SearchConfig(
        max_results_per_search=10,
        include_domains=["arxiv.org", "nature.com"],
    ),
)

app = build_app()

result = await app.ainvoke({
    "root_query": "Your research question",
    "config": config,
})


⸻

Project Structure

├── run.py                # CLI entry
├── benchmark.py          # Batch evaluation
├── sample_tasks.json
├── lighteningresearch/
│   ├── config.py         # Configuration system
│   ├── graph.py          # LangGraph orchestration
│   ├── nodes.py          # Planner/search/synth nodes
│   ├── tools.py          # Search tools
│   ├── memory.py         # Memory seeding
│   └── evaluation.py     # RACE + FACT scoring


⸻

If I had more time (next improvements)

Reliability
	•	retry logic + exponential backoff for searches
	•	better timeout handling
	•	partial failure recovery

Cost/latency optimization
	•	caching search results
	•	dynamic model routing
	•	early exit when confidence high

Better evaluation
	•	human preference scoring
	•	retrieval grounding benchmarks
	•	hallucination detection

Deployment
	•	FastAPI service wrapper
	•	streaming responses
	•	Docker image
	•	async queue for long research jobs

⸻

Why I built this

Most deep research agents today are sequential and slow.
I wanted to explore what happens when you treat research as a parallel, adaptive system instead of a prompt chain.

This project focuses on:
	•	orchestration design
	•	evaluation
	•	configurability
	•	real-world system tradeoffs

⸻

License

MIT

