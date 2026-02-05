"""
Dataset utilities for FlashResearch experiments.

Supports:
- Researchy Questions dataset (1,000 queries)
- FineWeb static corpus integration
- Sample questions for quick testing
"""

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class ResearchyQuestion:
    """A question from the Researchy Questions dataset."""
    id: str
    query: str
    field: str = "General"
    difficulty: str = "medium"
    expected_aspects: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Sample questions for testing (subset of typical Researchy Questions)
SAMPLE_QUESTIONS: List[ResearchyQuestion] = [
    ResearchyQuestion(
        id="rq_001",
        query="What are the current leading approaches to quantum error correction and how do they compare in terms of overhead and fault-tolerance thresholds?",
        field="Physics",
        difficulty="hard",
        expected_aspects=["surface codes", "topological codes", "overhead", "thresholds"],
    ),
    ResearchyQuestion(
        id="rq_002",
        query="How has CRISPR-Cas9 technology evolved since 2020, and what are the key regulatory and ethical debates surrounding its clinical applications?",
        field="Biology",
        difficulty="hard",
        expected_aspects=["base editing", "prime editing", "clinical trials", "ethics"],
    ),
    ResearchyQuestion(
        id="rq_003",
        query="What are the main architectural differences between transformer models and state space models, and what are the tradeoffs for long-context tasks?",
        field="AI/ML",
        difficulty="medium",
        expected_aspects=["attention", "Mamba", "S4", "context length", "efficiency"],
    ),
    ResearchyQuestion(
        id="rq_004",
        query="What is the current scientific consensus on the effectiveness of intermittent fasting for longevity, and what are the proposed mechanisms?",
        field="Health",
        difficulty="medium",
        expected_aspects=["autophagy", "caloric restriction", "clinical studies", "mechanisms"],
    ),
    ResearchyQuestion(
        id="rq_005",
        query="How do different carbon capture technologies compare in terms of cost, scalability, and energy requirements?",
        field="Environment",
        difficulty="medium",
        expected_aspects=["DAC", "BECCS", "ocean-based", "costs", "scalability"],
    ),
    ResearchyQuestion(
        id="rq_006",
        query="What are the leading theories explaining dark matter, and what experimental evidence supports or contradicts each?",
        field="Physics",
        difficulty="hard",
        expected_aspects=["WIMPs", "axions", "primordial black holes", "detection experiments"],
    ),
    ResearchyQuestion(
        id="rq_007",
        query="How do federated learning approaches handle non-IID data distributions, and what are the current solutions for communication efficiency?",
        field="AI/ML",
        difficulty="hard",
        expected_aspects=["FedAvg", "non-IID", "compression", "privacy"],
    ),
    ResearchyQuestion(
        id="rq_008",
        query="What are the neurobiological mechanisms of psychedelic-assisted therapy, and what does the clinical trial evidence show for depression treatment?",
        field="Neuroscience",
        difficulty="hard",
        expected_aspects=["psilocybin", "serotonin", "default mode network", "clinical trials"],
    ),
    ResearchyQuestion(
        id="rq_009",
        query="How do modern battery technologies compare for grid-scale energy storage, including cost, lifespan, and environmental impact?",
        field="Energy",
        difficulty="medium",
        expected_aspects=["lithium-ion", "solid-state", "flow batteries", "costs"],
    ),
    ResearchyQuestion(
        id="rq_010",
        query="What are the main approaches to AI alignment, and what are the open problems in ensuring AI systems remain beneficial?",
        field="AI Safety",
        difficulty="hard",
        expected_aspects=["RLHF", "constitutional AI", "interpretability", "scalable oversight"],
    ),
]


def load_researchy_questions(
    path: Optional[str] = None,
    limit: Optional[int] = None,
    field_filter: Optional[str] = None,
) -> List[ResearchyQuestion]:
    """
    Load Researchy Questions dataset.

    Args:
        path: Path to dataset JSON file. If None, uses sample questions.
        limit: Maximum number of questions to load
        field_filter: Only load questions from this field

    Returns:
        List of ResearchyQuestion objects

    Dataset format (JSON):
        [
            {
                "id": "rq_001",
                "query": "...",
                "field": "Physics",
                "difficulty": "hard",
                "expected_aspects": ["...", "..."]
            },
            ...
        ]
    """
    if path is None:
        # Use sample questions
        questions = SAMPLE_QUESTIONS.copy()
    else:
        # Load from file
        with open(path) as f:
            data = json.load(f)

        questions = []
        for item in data:
            q = ResearchyQuestion(
                id=item.get("id", f"rq_{len(questions):03d}"),
                query=item.get("query", item.get("question", "")),
                field=item.get("field", item.get("category", "General")),
                difficulty=item.get("difficulty", "medium"),
                expected_aspects=item.get("expected_aspects"),
            )
            questions.append(q)

    # Apply filters
    if field_filter:
        questions = [q for q in questions if q.field.lower() == field_filter.lower()]

    if limit:
        questions = questions[:limit]

    return questions


class FineWebCorpus:
    """
    Interface for FineWeb static corpus (for reproducibility).

    The FineWeb corpus provides a static set of web documents
    to ensure reproducible search results across experiments.

    Usage:
        corpus = FineWebCorpus("./fineweb_cache")
        results = corpus.search("quantum computing", max_results=10)
    """

    def __init__(self, corpus_dir: str):
        self.corpus_dir = Path(corpus_dir)
        self.index_file = self.corpus_dir / "index.json"
        self.index: Dict[str, Any] = {}

        if self.index_file.exists():
            with open(self.index_file) as f:
                self.index = json.load(f)

    def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search the static corpus.

        For full reproducibility, pre-cache all search results
        using the SearchCache from cache.py.
        """
        # Check if we have cached results for this query
        query_lower = query.lower().strip()

        # Simple keyword matching against cached documents
        results = []

        for doc_id, doc_meta in self.index.get("documents", {}).items():
            doc_path = self.corpus_dir / f"{doc_id}.json"
            if not doc_path.exists():
                continue

            with open(doc_path) as f:
                doc = json.load(f)

            # Simple relevance: count query terms in content
            content_lower = doc.get("content", "").lower()
            title_lower = doc.get("title", "").lower()

            query_terms = query_lower.split()
            matches = sum(1 for term in query_terms if term in content_lower or term in title_lower)

            if matches > 0:
                results.append({
                    "url": doc.get("url", ""),
                    "title": doc.get("title", ""),
                    "content": doc.get("content", "")[:500],
                    "score": matches / len(query_terms),
                })

        # Sort by score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]

    def add_document(
        self,
        url: str,
        title: str,
        content: str,
    ):
        """Add a document to the corpus."""
        import hashlib

        doc_id = hashlib.sha256(url.encode()).hexdigest()[:16]

        doc = {
            "url": url,
            "title": title,
            "content": content,
        }

        self.corpus_dir.mkdir(parents=True, exist_ok=True)

        with open(self.corpus_dir / f"{doc_id}.json", "w") as f:
            json.dump(doc, f, indent=2)

        if "documents" not in self.index:
            self.index["documents"] = {}

        self.index["documents"][doc_id] = {
            "url": url,
            "title": title[:100],
        }

        with open(self.index_file, "w") as f:
            json.dump(self.index, f, indent=2)

    def stats(self) -> Dict[str, Any]:
        """Get corpus statistics."""
        return {
            "total_documents": len(self.index.get("documents", {})),
            "corpus_dir": str(self.corpus_dir),
        }
