"""Fact-checker agent for Veritatis.

Accepts a natural-language claim and returns a truthfulness score (0-1)
based on metadata retrieved from Milvus tier collections.

Usage::

    from veritatis.agent import fact_check

    result = fact_check("A 7.5 magnitude earthquake struck Turkey in 2023")
    print(result.score, result.reasoning, result.sources)
"""

import json
import logging
import os
import re
from typing import Optional

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from veritatis.plain_search import vector_search

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — override via env-vars in production
# ---------------------------------------------------------------------------

_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://192.168.8.22:9999/v1")
_API_KEY = os.getenv("OPENAI_API_KEY", "api-key")
_MODEL = os.getenv("FACT_CHECK_MODEL", "gemma-4-31B-it-Q8_0")
_MAX_TOKENS = int(os.getenv("FACT_CHECK_MAX_TOKENS", "28000"))  # under n_ctx=29000

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
# Tools
# ---------------------------------------------------------------------------


@tool
def retrieve_tier1(claim: str) -> list[dict]:
    """Search tier1 (raw/unverified sources).

    Low weight in scoring — used as indirect confirmation.
    Returns: iid, credibility_score, date, domain, distance.
    """
    results = vector_search(claim, top_k=10, tier=1)
    logger.info(
        "retrieve_tier1 returned %d docs for claim: %r", len(results), claim[:60]
    )
    return results


@tool
def retrieve_tier2(claim: str) -> list[dict]:
    """Search tier2 (credible/verified sources).

    High weight in scoring — used as primary evidence.
    Returns: iid, credibility_score, date, domain, distance.
    """
    results = vector_search(claim, top_k=10, tier=2)
    logger.info(
        "retrieve_tier2 returned %d docs for claim: %r", len(results), claim[:60]
    )
    return results


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a general-purpose fact-checker. The domain is
determined by the data in the database, not limited to any specific topic.

You have access to two retrieval tools backed by a vector database:

  retrieve_tier1 — Tier 1 sources (raw, unverified).
    LOW weight: use as indirect confirmation only.
    credibility_score typically < 0.7.

  retrieve_tier2 — Tier 2 sources (credible, verified).
    HIGH weight: primary evidence for scoring.
    credibility_score >= 0.7.

Each tool returns metadata records (NOT article text):
  • iid              — unique record ID
  • credibility_score — source credibility 0.0–1.0
  • date             — event timestamp (epoch ms)
  • domain           — source website domain
  • distance         — COSINE similarity to the claim (0=unrelated, 1=identical)

MANDATORY PROCEDURE
───────────────────
Step 1 — ALWAYS call retrieve_tier1(claim)
Step 2 — ALWAYS call retrieve_tier2(claim)
Step 3 — If distance values are low (< 0.3), call one or both tools again with
          a rephrased query to improve coverage.
Step 4 — Produce a verdict based ONLY on retrieved metadata.

SCORING LOGIC
─────────────
  tier2 hit with distance >= 0.6  → strongly raises score
  tier1 hit without tier2          → weak confirmation, moderate score
  both empty OR distance < 0.3    → low score
  Score MUST be based solely on retrieved data, NOT pre-training knowledge.

SCORE SCALE
───────────
  0.8 – 1.0  strongly supported
  0.6 – 0.8  moderately supported
  0.4 – 0.6  uncertain
  0.2 – 0.4  weak support
  0.0 – 0.2  unsupported / contradicted"""

# ---------------------------------------------------------------------------
# JSON schema for final response
# ---------------------------------------------------------------------------

_JSON_SCHEMA = """{
  "score": <float 0.0-1.0>,
  "reasoning": "<concise explanation citing similarity and credibility numbers>",
  "sources": ["<domain1>", "<domain2>", ...]
}"""

# ---------------------------------------------------------------------------
# LLM + Agent singletons
# ---------------------------------------------------------------------------

_llm: Optional[ChatOpenAI] = None
_agent = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            base_url=_BASE_URL,
            api_key=_API_KEY,
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
        )
        logger.info(
            "Fact-checker LLM initialised (model=%s, base_url=%s)", _MODEL, _BASE_URL
        )
    return _llm


def _get_agent():
    global _agent
    if _agent is None:
        llm = _get_llm()
        _agent = create_react_agent(
            llm,
            tools=[retrieve_tier1, retrieve_tier2],
            state_modifier=_SYSTEM_PROMPT,
        )
        logger.info("ReAct agent initialised (model=%s)", _MODEL)
    return _agent


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
        parsed: dict = json.loads(match.group(1))
        return parsed
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        parsed = json.loads(match.group(0))
        return parsed
    raise ValueError(f"No JSON object found in LLM response: {text[:200]}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fact_check(claim: str) -> FactCheckResult:
    """Check the truthfulness of *claim* using a LangGraph ReAct agent.

    The agent calls retrieve_tier1 and retrieve_tier2, then produces a
    structured verdict based solely on the retrieved metadata.

    Args:
        claim: A natural-language statement to verify.

    Returns:
        FactCheckResult with score (0-1), reasoning, and source domains.
    """
    agent = _get_agent()
    user_msg = (
        f"CLAIM: {claim}\n\n"
        "After using the tools, respond with ONLY a valid JSON object:\n"
        f"{_JSON_SCHEMA}"
    )
    result = agent.invoke({"messages": [("user", user_msg)]})

    messages = result.get("messages", [])
    last_content = ""
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if content and isinstance(content, str):
            last_content = content
            break

    logger.debug("Agent final response: %s", last_content[:300])
    data = _parse_json_response(last_content)
    return FactCheckResult(
        score=float(data["score"]),
        reasoning=str(data["reasoning"]),
        sources=list(data.get("sources", [])),
    )
