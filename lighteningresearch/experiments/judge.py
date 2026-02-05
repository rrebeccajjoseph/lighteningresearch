"""
LLM Judge for scoring research reports.

Implements the scoring rubric from the FlashResearch paper:
- Quality: Organization and clarity (0-100)
- Relevance: How well it answers the user's prompt (0-100)
- Faithfulness: Absence of hallucinations (0-100)
"""

import json
import re
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class JudgeScores:
    """Scores from the LLM judge."""
    quality: float          # Organization and clarity
    relevance: float        # How well it answers the prompt
    faithfulness: float     # Absence of hallucinations
    overall: float          # Weighted average
    explanation: str        # Judge's reasoning

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


JUDGE_SYSTEM_PROMPT = """You are an expert research report evaluator. Your task is to score a research report on three dimensions.

## Scoring Rubric (0-100 for each)

### Quality (Organization & Clarity)
- 90-100: Exceptionally well-organized with clear sections, logical flow, proper citations, and professional formatting
- 70-89: Well-structured with clear main points, some minor organizational issues
- 50-69: Adequate organization but lacks clear structure or has formatting issues
- 30-49: Poorly organized, difficult to follow, missing key sections
- 0-29: Disorganized, incoherent, or incomplete

### Relevance (Answers the Query)
- 90-100: Directly and comprehensively addresses the query with depth and nuance
- 70-89: Addresses the main question well with some gaps in coverage
- 50-69: Partially addresses the query but misses important aspects
- 30-49: Only tangentially related to the query
- 0-29: Does not address the query or is off-topic

### Faithfulness (No Hallucinations)
- 90-100: All claims are supported by the provided context, no fabricated information
- 70-89: Most claims are supported, minor unsupported details
- 50-69: Some unsupported claims but core information is accurate
- 30-49: Multiple hallucinations or fabricated facts
- 0-29: Significant hallucinations, unreliable information

## Output Format
You MUST respond with valid JSON in this exact format:
{
    "quality": <0-100>,
    "relevance": <0-100>,
    "faithfulness": <0-100>,
    "explanation": "<brief explanation of scores>"
}"""


JUDGE_USER_TEMPLATE = """## Original Research Query
{query}

## Gathered Context (Source Material)
{context}

## Generated Report
{report}

---
Score this report on Quality, Relevance, and Faithfulness.
Respond with JSON only."""


class LLMJudge:
    """
    LLM-based judge for evaluating research reports.

    Usage:
        judge = LLMJudge()
        scores = await judge.score(query, report, context)
        print(f"Quality: {scores.quality}, Relevance: {scores.relevance}")
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.1,
    ):
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    async def score(
        self,
        query: str,
        report: str,
        context: Optional[str] = None,
        findings: Optional[List[Dict[str, Any]]] = None,
    ) -> JudgeScores:
        """
        Score a research report.

        Args:
            query: The original research query
            report: The generated research report
            context: Optional raw context string
            findings: Optional list of findings (will be formatted as context)

        Returns:
            JudgeScores with quality, relevance, faithfulness scores
        """
        # Build context from findings if not provided directly
        if context is None and findings:
            context_parts = []
            for f in findings[:20]:  # Limit to avoid token overflow
                if isinstance(f, dict):
                    title = f.get("title", "Untitled")
                    url = f.get("url", "")
                    content = f.get("content", "")
                else:
                    title = getattr(f, "title", "Untitled")
                    url = getattr(f, "url", "")
                    content = getattr(f, "content", "")

                content = (content or "")[:500]
                context_parts.append(f"### {title}\nSource: {url}\n{content}")
            context = "\n\n".join(context_parts)
        elif context is None:
            context = "[No context provided - judge based on report alone]"

        # Truncate if needed
        if len(context) > 15000:
            context = context[:15000] + "\n...[truncated]"
        if len(report) > 10000:
            report = report[:10000] + "\n...[truncated]"

        user_prompt = JUDGE_USER_TEMPLATE.format(
            query=query,
            context=context,
            report=report,
        )

        messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = await self.llm.ainvoke(messages)

        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())

            quality = float(data.get("quality", 50))
            relevance = float(data.get("relevance", 50))
            faithfulness = float(data.get("faithfulness", 50))

            # Calculate overall (equal weighting as per paper)
            overall = (quality + relevance + faithfulness) / 3

            return JudgeScores(
                quality=quality,
                relevance=relevance,
                faithfulness=faithfulness,
                overall=overall,
                explanation=data.get("explanation", ""),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback: try to extract scores from text
            return self._parse_fallback(response.content)

    def _parse_fallback(self, content: str) -> JudgeScores:
        """Attempt to parse scores from non-JSON response."""
        quality = relevance = faithfulness = 50.0

        # Try to find numbers after keywords
        patterns = [
            (r"quality[:\s]+(\d+)", "quality"),
            (r"relevance[:\s]+(\d+)", "relevance"),
            (r"faithfulness[:\s]+(\d+)", "faithfulness"),
        ]

        for pattern, field in patterns:
            match = re.search(pattern, content.lower())
            if match:
                value = float(match.group(1))
                if field == "quality":
                    quality = value
                elif field == "relevance":
                    relevance = value
                elif field == "faithfulness":
                    faithfulness = value

        overall = (quality + relevance + faithfulness) / 3

        return JudgeScores(
            quality=quality,
            relevance=relevance,
            faithfulness=faithfulness,
            overall=overall,
            explanation="[Parsed from non-JSON response]",
        )

    def score_sync(
        self,
        query: str,
        report: str,
        context: Optional[str] = None,
        findings: Optional[List[Dict[str, Any]]] = None,
    ) -> JudgeScores:
        """Synchronous version of score()."""
        import asyncio
        return asyncio.run(self.score(query, report, context, findings))


async def score_report(
    query: str,
    report: str,
    context: Optional[str] = None,
    findings: Optional[List[Dict[str, Any]]] = None,
    model: str = "gpt-4o",
) -> JudgeScores:
    """
    Convenience function to score a report.

    Args:
        query: Original research query
        report: Generated report
        context: Optional context string
        findings: Optional list of findings
        model: Judge model to use

    Returns:
        JudgeScores
    """
    judge = LLMJudge(model=model)
    return await judge.score(query, report, context, findings)
