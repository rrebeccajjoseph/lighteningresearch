# LightningResearch

LightningResearch is a LangGraph-based deep research agent that plans, searches, and synthesizes grounded research reports using parallel web search and adaptive reasoning.

The goal of this project is to demonstrate how a research agent can move beyond simple sequential prompting, and now has parallel orchestration, adaptive planning, and evaluation for real-world LLM systems.

All experiments are run using the FlashResearch architecture — a planner-orchestrated, parallel research system that decomposes queries into subquestions, executes concurrent search workers, and synthesizes results under a fixed time budget.

---

# What this system does

Given a research question, LightningResearch:

1. Breaks the query into sub-questions  
2. Runs parallel web searches  
3. Scores relevance and quality using LLMs  
4. Explores promising leads more deeply  
5. Synthesizes a structured research report  
6. Evaluates output quality automatically  

The system is built on LangGraph, enabling dynamic routing, concurrency, and stateful orchestration rather than a static DAG.

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

Testing (short)
Run all tests:
pytest

Fast unit-only subset:
pytest tests/unit/test_config.py tests/unit/test_models.py tests/unit/test_cache.py

What the unit tests cover:
- Config loading/validation and env handling
- Core data models (state, tasks, findings)
- Search cache behavior and reproducible corpus

Integration tests:
pytest tests/integration/test_evaluation.py tests/integration/test_baselines.py tests/integration/test_experiments.py

⸻

Configuration Presets

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

Run a single evaluation:
python run.py --eval "Your query"

All experiments use the FlashResearch architecture and are scored using two frameworks:
- RACE: evaluates comprehensiveness, depth, instruction following, and readability
- FACT: evaluates citation accuracy, efficiency, and source verification

Table 1 — Throughput vs. Quality:
Runs `run_experiments.py --table1` to compare LightningResearch report quality under strict
time budgets (default 2 vs 10 minutes), optionally including a baseline.

Figure 2 — Breadth/Depth trade-off:
Runs `run_experiments.py --figure2` to vary LightningResearch breadth/depth and see where
quality saturates under RACE+FACT scoring.

Quick Test — Sanity check:
Runs `run_experiments.py --quick-test` to validate orchestration, scoring, and output
formatting on a small batch before full experiments.

Benchmarking (batch regression):
Uses `benchmark.py` to run a task file with a chosen time budget and optionally compare
against baselines.



Save results:
python run.py --eval --output results.json "Your query"



Quiet mode (Prints only final report!):
python run.py --quiet "Your query"


⸻

## Python Usage 
You don't just have to use a terminal arg command - you can also use LightningResearch as a library.

from lighteningresearch import AgentConfig, ModelConfig, SearchConfig, build_app, build_initial_state

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
state = build_initial_state("Your research question", config)
result = await app.ainvoke(state)

⸻

Project Structure

├── run.py                # CLI entry (shim)
├── benchmark.py          # Benchmark CLI (shim)
├── run_experiments.py    # Experiments CLI (shim)
├── sample_tasks.json
├── lighteningresearch/
│   ├── config.py         # Configuration system
│   ├── configs/          # Preset configurations
│   ├── graph.py          # LangGraph orchestration
│   ├── nodes.py          # Planner/search/synth nodes
│   ├── tools.py          # Search tools
│   ├── state_builder.py  # Shared initial state builder
│   ├── memory.py         # Memory seeding
│   ├── evaluation.py     # RACE + FACT scoring
│   └── cli/              # CLI implementations
├── tests/
│   ├── unit/
│   └── integration/


Why I built this

Most deep research agents today are sequential and slow.
I wanted to explore what happens when you treat research as an adaptive system instead of a prompt chain.

This project focuses on:
	•	orchestration design
	•	evaluation
	•	configurability/real-world system tradeoffs
