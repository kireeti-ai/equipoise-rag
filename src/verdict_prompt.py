"""
verdict_prompt.py
Three prompt variants for INV-03 prompt sensitivity investigation.

NEUTRAL:    asks LLM to summarise evidence without structure
BIASED:     asks only for supporting evidence -- deliberately one-sided
STRUCTURED: asks for explicit sections with citations -- most grounded

Switch variant by changing PROMPT_VARIANT in config.py.
"""

from src.config import PROMPT_VARIANT


NEUTRAL_PROMPT = """You are a biomedical evidence synthesis expert.

STEP 1: EVIDENCE FILTERING
Before writing your summary, review the abstracts. Identify which abstracts (by PMID) are DIRECTLY about the core subject of the claim (e.g., if the claim is about "Depression", the abstract must mention depression or clinical mood disorders).
- If an abstract is only about a mechanism (e.g., inflammation) or a different condition (e.g., arthritis), it is IRRELEVANT. 
- If ZERO abstracts are directly relevant, your only output must be: "No directly relevant evidence found in the retrieved abstracts."

STEP 2: SUMMARY
Only if you found relevant abstracts in Step 1, provide an objective summary of the evidence for and against the claim. 
- DO NOT use speculative language ("could", "may", "potential") to bridge gaps.
- DO NOT infer that a general process applies to a specific disease unless explicitly stated.

Claim: {claim}

Retrieved abstracts:
{abstracts}
"""


BIASED_PROMPT = """You are a biomedical evidence synthesis expert.

Claim: {claim}

Retrieved abstracts:
{abstracts}

Based on the retrieved abstracts, what evidence supports this claim?
Summarise the supporting findings and state how confident you are.
"""


STRUCTURED_PROMPT = """You are a biomedical evidence synthesis expert.

CRITICAL RULES (ANTI-HALLUCINATION):
1. RELEVANCE CHECK: If an abstract does not mention the EXACT core subject of the claim, it must be DISQUALIFIED.
2. NO SPECULATION: Do NOT "stretch" mechanism papers (e.g., cell studies) to support clinical claims.
3. If ZERO abstracts are directly about the claim's core condition, you MUST select "Insufficient Evidence" for the VERDICT.
4. Do not begin your response with any preamble. Start directly with VERDICT.
5. Reference evidence using [1], [2], [3] inline, matching the numbered citations below.
6. Do NOT output any HTML or markdown formatting tags. Output plain text only.
7. You MUST acknowledge refuting evidence. Do not omit or minimize it.

Claim: {claim}

COMPUTED CONFIDENCE: {confidence_label} (Agreement Score: {confidence_score})
  Supporting studies: {supports_count} | Refuting studies: {refutes_count} | Neutral: {neutral_count}

--- SUPPORTING EVIDENCE ---
{supporting_abstracts}

--- REFUTING EVIDENCE ---
{refuting_abstracts}

--- NEUTRAL / BACKGROUND EVIDENCE ---
{neutral_abstracts}

Produce your verdict in exactly this format:

VERDICT: [Supported / Contradicted / Mixed / Insufficient Evidence]

SUMMARY:
[2-3 sentence plain English summary with inline [1][2][3] citations]

SUPPORTING EVIDENCE:
[Cite specific findings that support the claim, or "None found"]

CONTRADICTING EVIDENCE:
[Cite specific findings that contradict the claim, or "None found"]

CONFIDENCE: {confidence_label} — [one sentence explaining why, referencing the evidence balance above]
"""


PROMPT_VARIANTS = {
    "neutral": NEUTRAL_PROMPT,
    "biased": BIASED_PROMPT,
    "structured": STRUCTURED_PROMPT
}


def build_verdict_prompt(claim: str, retrieved: list, variant: str = PROMPT_VARIANT, confidence_info: dict = None) -> str:
    """
    Builds the full prompt from claim and retrieved abstracts.
    For the 'structured' variant, groups evidence by stance (SUPPORTS/REFUTES/NEUTRAL).

    Args:
        claim:           the input biomedical claim
        retrieved:       list of abstract dicts from retriever/reranker (may have 'stance' key)
        variant:         which prompt variant to use (neutral/biased/structured)
        confidence_info: dict from compute_confidence() — used in structured prompt

    Returns:
        formatted prompt string ready to send to the LLM
    """
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant: {variant}. Choose: neutral | biased | structured")

    if variant == "structured" and confidence_info is not None:
        # Group evidence by stance for contradiction-aware synthesis
        supporting, refuting, neutral = [], [], []
        for r in retrieved:
            entry = (f"[{r['id']}] {r['title']}\n{r['text'][:500]}..."
                     if len(r['text']) > 500 else f"[{r['id']}] {r['title']}\n{r['text']}")
            stance = r.get("stance", "NEUTRAL")
            if stance == "SUPPORTS":
                supporting.append(entry)
            elif stance == "REFUTES":
                refuting.append(entry)
            else:
                neutral.append(entry)

        def fmt(lst):
            return "\n\n".join(lst) if lst else "None found"

        return STRUCTURED_PROMPT.format(
            claim=claim,
            confidence_label=confidence_info.get("confidence_label", "Unknown"),
            confidence_score=confidence_info.get("confidence_score", 0.0),
            supports_count=confidence_info.get("supports_count", 0),
            refutes_count=confidence_info.get("refutes_count", 0),
            neutral_count=confidence_info.get("neutral_count", 0),
            supporting_abstracts=fmt(supporting),
            refuting_abstracts=fmt(refuting),
            neutral_abstracts=fmt(neutral),
        )

    # Fallback for non-structured variants: flat numbered list
    abstracts_text = ""
    for i, r in enumerate(retrieved):
        abstracts_text += f"[{i+1}] PMID: {r['id']}\n"
        abstracts_text += f"Title: {r['title']}\n"
        abstracts_text += f"Abstract: {r['text']}\n\n"

    template = PROMPT_VARIANTS[variant]
    return template.format(
        claim=claim,
        abstracts=abstracts_text.strip()
    )


if __name__ == "__main__":
    sample_retrieved = [
        {
            "id": "12345678",
            "title": "Omega-3 and depression meta-analysis",
            "text": "We found no significant effect of omega-3 on depression scores in 500 adults."
        },
        {
            "id": "87654321",
            "title": "Omega-3 deficiency linked to depressive symptoms",
            "text": "Low omega-3 levels were significantly associated with higher rates of clinical depression."
        }
    ]

    claim = "Omega-3 supplementation reduces symptoms of depression"

    for variant in ["neutral", "biased", "structured"]:
        print(f"\n{'='*50}")
        print(f"VARIANT: {variant.upper()}")
        print('='*50)
        prompt = build_verdict_prompt(claim, sample_retrieved, variant=variant)
        print(prompt)
