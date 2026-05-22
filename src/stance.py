"""
stance.py
Evidence stance classification for biomedical claim verification.

For each retrieved abstract, classifies whether it:
  - SUPPORTS  the claim
  - REFUTES   the claim
  - NEUTRAL   (not directly relevant)

Uses the existing LLM client (OpenRouter / Groq fallback).
"""

import os
from openai import OpenAI
from groq import Groq
from src.config import GROQ_API_KEY, GROQ_REFORMULATOR_MODEL


def classify_stance(claim: str, chunk_text: str, title: str = "") -> str:
    """
    Classifies the stance of a single evidence chunk relative to a claim.

    Args:
        claim:      biomedical claim in plain English
        chunk_text: abstract passage text
        title:      paper title (adds context)

    Returns:
        One of "SUPPORTS", "REFUTES", or "NEUTRAL"
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
        )
    else:
        client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""You are a biomedical NLI (Natural Language Inference) expert.

Given the following biomedical claim and a scientific abstract passage, determine whether the passage:
- SUPPORTS  the claim (evidence in favor)
- REFUTES   the claim (evidence against or contradicts)
- NEUTRAL   (passage is not directly about the claim, or is inconclusive)

Rules:
- Respond with EXACTLY one word: SUPPORTS, REFUTES, or NEUTRAL
- Do NOT explain your answer
- Animal/in-vitro studies that cannot be extrapolated to humans → NEUTRAL
- Mechanistic papers without clinical outcome data → NEUTRAL

Claim: {claim}

Paper Title: {title}
Abstract Passage: {chunk_text[:600]}

Stance:"""

    try:
        response = client.chat.completions.create(
            model=GROQ_REFORMULATOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5
        )
        result = response.choices[0].message.content.strip().upper()
        # Sanitize — only accept valid labels
        if result in ("SUPPORTS", "REFUTES", "NEUTRAL"):
            return result
        # Fallback: try to parse partial responses
        for label in ("SUPPORTS", "REFUTES", "NEUTRAL"):
            if label in result:
                return label
        return "NEUTRAL"
    except Exception as e:
        print(f"  [stance] Error: {e}")
        return "NEUTRAL"


def classify_stance_batch(claim: str, chunks: list) -> list:
    """
    Classifies stance for a list of retrieved chunks.

    Args:
        claim:  biomedical claim
        chunks: list of dicts with keys: text, title

    Returns:
        The same list of dicts, each augmented with a "stance" key.
    """
    print(f"\n--- Stance Classification ---")
    for chunk in chunks:
        stance = classify_stance(claim, chunk.get("text", ""), chunk.get("title", ""))
        chunk["stance"] = stance
        label_icon = {"SUPPORTS": "✅", "REFUTES": "❌", "NEUTRAL": "⬜"}.get(stance, "?")
        print(f"  {label_icon} [{stance}] {chunk.get('title', 'Unknown')[:70]}")
    print(f"----------------------------\n")
    return chunks


def compute_confidence(chunks: list) -> dict:
    """
    Computes a structured confidence score based on stance distribution.

    Agreement Ratio = |Supports - Refutes| / (Supports + Refutes + 0.01)
    Final Confidence = Agreement_Ratio * Avg_Rerank_Score

    Returns:
        dict with: supports_count, refutes_count, neutral_count,
                   agreement_ratio, confidence_score, confidence_label
    """
    stances = [c.get("stance", "NEUTRAL") for c in chunks]
    supports = stances.count("SUPPORTS")
    refutes  = stances.count("REFUTES")
    neutral  = stances.count("NEUTRAL")

    agreement_ratio = abs(supports - refutes) / (supports + refutes + 1e-6)

    avg_rerank_score = 0.5
    rerank_scores = [c.get("rerank_score", c.get("score", 0.5)) for c in chunks if c.get("stance") != "NEUTRAL"]
    if rerank_scores:
        avg_rerank_score = sum(rerank_scores) / len(rerank_scores)

    confidence_score = agreement_ratio * avg_rerank_score

    if confidence_score >= 0.6:
        label = "High"
    elif confidence_score >= 0.3:
        label = "Medium"
    else:
        label = "Low"

    # Override to "Controversial" when there is substantial disagreement
    if supports >= 2 and refutes >= 2:
        label = "Controversial"

    return {
        "supports_count": supports,
        "refutes_count": refutes,
        "neutral_count": neutral,
        "agreement_ratio": round(agreement_ratio, 3),
        "confidence_score": round(confidence_score, 3),
        "confidence_label": label
    }
