"""
pipeline.py
Full end-to-end RAG pipeline.
Takes a claim, runs retrieval, reranking, and verdict generation.
Returns a structured verdict with citations.
"""

from groq import Groq
from src.config import (
    GROQ_API_KEY, GROQ_VERDICT_MODEL, RETRIEVAL_METHOD,
    RETRIEVAL_CANDIDATE_K, TOP_K, PROMPT_VARIANT
)
from src.retriever import retrieve
from src.reranker import rerank
from src.reformulator import reformulate_query
from src.verdict_prompt import build_verdict_prompt
from src.stance import classify_stance_batch, compute_confidence


def run_pipeline(claim: str, method: str = RETRIEVAL_METHOD) -> dict:
    """
    Runs the full Equipoise pipeline on a single claim.

    Steps:
    1. If method is queryreform, reformulate the claim first
    2. Retrieve relevant abstracts using the configured strategy
    3. Rerank retrieved abstracts with cross-encoder
    4. Build structured verdict prompt
    5. Generate verdict using Groq LLM
    6. Return structured result

    Args:
        claim: biomedical claim in plain English
        method: retrieval strategy to use

    Returns:
        dict with claim, method, retrieved abstracts, and verdict
    """
    print(f"Running pipeline | claim: {claim[:60]}... | method: {method}")

    # Step 1: query reformulation if needed
    reformulated = None
    if method == "queryreform":
        print("Reformulating query...")
        reformulated = reformulate_query(claim)
        print(f"Reformulated: {reformulated[:80]}...")

    # Step 2: retrieve
    print("Retrieving abstracts...")
    retrieved = retrieve(
        claim,
        method=method,
        reformulated_query=reformulated,
        top_k=RETRIEVAL_CANDIDATE_K
    )
    print(f"Retrieved {len(retrieved)} candidate abstracts")

    # Step 3: rerank
    print("Reranking...")
    reranked = rerank(claim, retrieved, top_k=TOP_K)
    print(f"Reranked to top {len(reranked)}")

    # Step 3b: stance classification
    print("Classifying evidence stance...")
    reranked = classify_stance_batch(claim, reranked)
    confidence_info = compute_confidence(reranked)
    print(f"Confidence: {confidence_info['confidence_label']} "
          f"(Score: {confidence_info['confidence_score']:.2f}) "
          f"| Supports: {confidence_info['supports_count']} "
          f"Refutes: {confidence_info['refutes_count']} "
          f"Neutral: {confidence_info['neutral_count']}")

    # Step 4: build prompt
    prompt = build_verdict_prompt(claim, reranked, variant=PROMPT_VARIANT, confidence_info=confidence_info)

    # Step 5: generate verdict
    print("Generating verdict...")
    
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
        
    response = client.chat.completions.create(
        model=GROQ_VERDICT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000
    )
    verdict_text = response.choices[0].message.content.strip()

    # Step 6: return result
    return {
        "claim": claim,
        "method": method,
        "prompt_variant": PROMPT_VARIANT,
        "retrieval_candidate_k": RETRIEVAL_CANDIDATE_K,
        "final_top_k": TOP_K,
        "reformulated_query": reformulated,
        "retrieved": reranked,
        "confidence": confidence_info,
        "verdict": verdict_text
    }


if __name__ == "__main__":
    test_claim = "Omega-3 fatty acids reduce symptoms of depression"

    result = run_pipeline(test_claim, method="bm25")

    print("\n" + "="*60)
    print("CLAIM:", result["claim"])
    print("METHOD:", result["method"])
    print("="*60)
    print(result["verdict"])
    print("="*60)
