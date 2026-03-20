"""Fact-checker agent for Veritatis.

Accepts a natural-language claim and returns a truthfulness score (0-1)
based on metadata retrieved from the Tier 2 Milvus collection.

Note: Milvus stores only embeddings + metadata (iid, credibility_score,
date, domain).  The agent reasons over those signals rather than raw text.

Usage::

    from veritatis.agent import fact_check

    result = fact_check("A 7.5 magnitude earthquake struck Turkey in 2023")
    print(result.score, result.reasoning, result.sources)
"""

import json
import logging
import os
import re
from typing import Any

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from veritatis.plain_search import vector_search

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — override via env-vars in production
# ---------------------------------------------------------------------------

_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://192.168.8.22:9999/v1")
_API_KEY = os.getenv("OPENAI_API_KEY", "api-key")
_MODEL = os.getenv("FACT_CHECK_MODEL", "gpt-4")
_MAX_TOKENS = int(os.getenv("FACT_CHECK_MAX_TOKENS", "80000"))
_MAX_ITERATIONS = 5

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class FactCheckResult(BaseModel):
    """Structured output from the fact-checker agent."""

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Truthfulness score from 0.0 (false / unsupported) "
            "to 1.0 (well-supported / likely true)"
        ),
    )
    reasoning: str = Field(
        ...,
        description="Step-by-step explanation of why this score was assigned",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="List of source domains retrieved from the database",
    )


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


def query_collection(claim: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Return top-k metadata records from the Tier 2 Milvus collection."""
    return vector_search(claim, top_k=top_k, tier=2)


@tool
def retrieve_facts(claim: str) -> list[dict]:
    """Retrieve up to 10 relevant earthquake records from the credible (Tier 2) database.

    Returns a list of metadata dicts with fields:
        iid, credibility_score, date, domain, distance (COSINE similarity 0-1).

    Use this tool to find supporting or contradicting evidence for a claim.
    Call it again with a rephrased query if the first results are sparse or
    have low similarity scores.
    """
    docs = query_collection(claim)
    logger.info("retrieve_facts returned %d docs for claim: %r", len(docs), claim[:60])
    return docs


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a fact-checker specialising in earthquake events.

You have access to a vector database (Tier 2 — credible sources) via the
`retrieve_facts` tool.  The database does NOT store article text; it stores
metadata for each indexed document:

  • iid              — unique record ID
  • credibility_score — source credibility rated 0.0-1.0 (Tier 2 >= 0.7)
  • date             — event timestamp (epoch ms)
  • domain           — source website domain
  • distance         — COSINE similarity between the claim and the document
                       (0.0 = unrelated, 1.0 = identical meaning)

HOW TO ASSESS A CLAIM
─────────────────────
1. Call `retrieve_facts` with the claim (and optionally with rephrased queries
   if the first batch has low `distance` values or is empty).
2. Examine the retrieved metadata:
   • High `distance` (≥ 0.6) + high `credibility_score` → strong evidence.
   • Many distinct `domain` values → multiple independent sources support it.
   • Sparse results or low `distance` (< 0.3) → claim is poorly supported.
   • Very low `distance` across ALL results → likely outside the knowledge base.
3. Produce a `score` in [0.0, 1.0]:
   • 0.8 – 1.0  Strongly supported by multiple credible, highly-similar records.
   • 0.6 – 0.8  Moderately supported.
   • 0.4 – 0.6  Uncertain — mixed signals or sparse coverage.
   • 0.2 – 0.4  Weak support; few or low-similarity records.
   • 0.0 – 0.2  Contradicted or entirely unsupported.
4. List the `domain` values of the top contributing records in `sources`.
5. Write a concise `reasoning` that cites similarity and credibility numbers.

IMPORTANT: base your score ONLY on the retrieved metadata, not on your
pre-training knowledge about earthquakes."""


# ---------------------------------------------------------------------------
# LLM singleton
# ---------------------------------------------------------------------------

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            base_url=_BASE_URL,
            api_key=_API_KEY,
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
        )
        logger.info("Fact-checker LLM initialised (model=%s, base_url=%s)", _MODEL, _BASE_URL)
    return _llm


def _extract_final_content(text: str) -> str:
    """Extract the final message content from channel-formatted LLM output.

    Some local LLMs return responses in a channel format:
        <|channel|>analysis<|message|>...<|end|>
        <|start|>assistant<|channel|>final<|message|>...

    We want only the 'final' channel content.
    """
    marker = "<|channel|>final<|message|>"
    if marker in text:
        return text.split(marker, 1)[1]
    # Fallback: strip any remaining channel tokens
    text = re.sub(r"<\|[^|]+\|>(?:analysis|final|message)?", "", text)
    return text.strip()


def _parse_json_response(text: str) -> dict:
    """Extract and parse a JSON object from LLM response text."""
    text = _extract_final_content(text)
    # Try to find a JSON block (```json ... ``` or bare {...})
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON object found in LLM response: {text[:200]}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_JSON_SCHEMA = """{
  "score": <float 0.0-1.0>,
  "reasoning": "<concise explanation citing similarity and credibility numbers>",
  "sources": ["<domain1>", "<domain2>", ...]
}"""


def fact_check(claim: str) -> FactCheckResult:
    """Check the truthfulness of *claim* against the Tier 2 Milvus collection.

    Retrieves relevant records directly (no tool-calling required from the LLM),
    then asks the LLM to reason over the metadata and return a structured verdict.

    Args:
        claim: A natural-language statement to verify.

    Returns:
        FactCheckResult with score (0-1), reasoning, and source domains.
    """
    # Step 1: retrieve evidence directly — no LLM tool-calling needed
    records = query_collection(claim)
    logger.info("Retrieved %d records for claim: %r", len(records), claim[:60])

    records_text = json.dumps(records, indent=2) if records else "[]"

    # Step 2: ask LLM to reason over the retrieved metadata and return raw JSON
    llm = _get_llm()
    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"CLAIM: {claim}\n\n"
        f"RETRIEVED RECORDS FROM DATABASE:\n{records_text}\n\n"
        "Based solely on the records above, produce your verdict.\n"
        f"Respond with ONLY a valid JSON object in this exact format:\n{_JSON_SCHEMA}"
    )
    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    logger.debug("LLM raw response: %s", raw[:300])

    data = _parse_json_response(raw)
    return FactCheckResult(
        score=float(data["score"]),
        reasoning=str(data["reasoning"]),
        sources=list(data.get("sources", [])),
    )
