"""
reformulator.py
Rewrites input claim using Groq LLM to include negation and null-result language.
This improves contradiction retrieval coverage for the queryreform strategy.
"""

from groq import Groq
from src.config import GROQ_API_KEY, GROQ_REFORMULATOR_MODEL


def reformulate_query(claim: str) -> str:
    """
    Takes a biomedical claim and returns a reformulated query
    that includes both confirming and disconfirming language.
    This helps dense retrieval surface contradicting abstracts.
    """
    import os
    from openai import OpenAI
    from groq import Groq
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
        )
    else:
        client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""You are a biomedical literature search expert.

Your task: rewrite the claim below into a search query that will retrieve BOTH supporting and contradicting scientific abstracts.

Rules:
- Include the original claim language
- Add negation variants: "no effect", "no significant effect", "failed to show", "did not improve", "null result"
- Add uncertainty language: "conflicting evidence", "inconsistent findings"
- Keep it under 60 words
- Return only the reformulated query, nothing else

Claim: {claim}

Reformulated query:"""

    response = client.chat.completions.create(
        model=GROQ_REFORMULATOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150
    )

    return response.choices[0].message.content.strip()

import json

def decompose_claim(claim: str) -> dict:
    """
    Decomposes a biomedical claim into a structured PICO + relation tuple.

    Returns:
        {
          "intervention": list[str],   # drug/treatment + synonyms
          "outcome":      list[str],   # health outcome + synonyms
          "population":   str,         # "humans" | "animal" | "general"
          "relation":     str,         # raw predicate ("cures", "prevents", ...)
          "relation_type": str         # treatment | prevention | causation |
                                       # association | prognosis | mechanism
        }
    """
    import os
    from openai import OpenAI
    from groq import Groq

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
    else:
        client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""You are a biomedical NLP expert. Decompose the following claim into its core components.

Return ONLY a valid JSON object with these exact keys:
- "intervention": list of strings (the drug/substance/exposure and its synonyms)
- "outcome": list of strings (the health outcome and synonyms)
- "population": string — one of: "humans", "animal", "general"
- "relation": string — the exact predicate verb from the claim (e.g. "cures", "prevents", "increases")
- "relation_type": string — one of: "treatment", "prevention", "causation", "association", "prognosis", "mechanism"

Relation type guide:
  treatment   → claim says X treats/cures/improves/reduces symptoms of Y
  prevention  → claim says X prevents/reduces risk of/protects against Y
  causation   → claim says X causes/induces/leads to Y
  association → claim says X is associated with / linked to Y
  prognosis   → claim says X predicts / affects survival / mortality of Y
  mechanism   → claim says X acts via / modulates / inhibits pathway of Y

Claim: {claim}

JSON Output:"""

    try:
        response = client.chat.completions.create(
            model=GROQ_REFORMULATOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300
        )

        content = response.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()

        data = json.loads(content)

        # Normalise + ensure keys exist
        for key in ["intervention", "outcome"]:
            if key not in data or not isinstance(data[key], list):
                data[key] = []
        data.setdefault("population", "humans")
        data.setdefault("relation", "")
        data.setdefault("relation_type", "association")

        # Validate relation_type
        valid_types = {"treatment", "prevention", "causation", "association", "prognosis", "mechanism"}
        if data["relation_type"] not in valid_types:
            data["relation_type"] = "association"

        return data

    except Exception as e:
        print(f"Decomposition error: {e}")
        return {
            "intervention": [], "outcome": [], "population": "humans",
            "relation": "", "relation_type": "association"
        }



if __name__ == "__main__":
    test_claims = [
        "Vitamin D supplementation improves depression symptoms",
        "Omega-3 fatty acids reduce cardiovascular disease risk",
        "Intermittent fasting is more effective than caloric restriction for weight loss"
    ]

    for claim in test_claims:
        print(f"Original:     {claim}")
        reformulated = reformulate_query(claim)
        print(f"Reformulated: {reformulated}")
        print()
