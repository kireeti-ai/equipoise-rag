"""
retriever.py
Implements all four retrieval strategies:
- dense: ChromaDB vector similarity search
- bm25: keyword frequency search via rank-bm25
- hybrid: reciprocal rank fusion of dense and BM25
- queryreform: LLM rewrites claim, then dense search
"""

import pickle
import torch
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from src.reformulator import decompose_claim
from src.config import (
    EMBEDDING_MODEL, CHROMA_DB_PATH, CHROMA_COLLECTION_NAME,
    TOP_K, HYBRID_DENSE_WEIGHT, HYBRID_BM25_WEIGHT, RETRIEVAL_METHOD
)

_cache = {}

# ──────────────────────────────────────────────
# Relation-aware query templates
# ──────────────────────────────────────────────
RELATION_QUERY_TEMPLATES = {
    "treatment": [
        "{intervention} treatment {outcome} clinical trial",
        "treating {outcome} with {intervention} randomized",
        "{intervention} therapy {outcome} efficacy",
    ],
    "prevention": [
        "{intervention} prevention {outcome}",
        "{intervention} reduces risk {outcome} cohort",
        "{intervention} prophylaxis {outcome} prospective",
    ],
    "causation": [
        "{intervention} causes {outcome}",
        "{intervention} increases risk {outcome} incidence",
        "association {intervention} {outcome} cohort study",
    ],
    "association": [
        "{intervention} associated {outcome}",
        "{intervention} correlation {outcome}",
        "{intervention} linked {outcome} observational",
    ],
    "prognosis": [
        "{outcome} prognosis {intervention} survival",
        "{intervention} mortality {outcome}",
        "{intervention} predictor {outcome} outcome",
    ],
    "mechanism": [
        "{intervention} mechanism {outcome} pathway",
        "{intervention} molecular {outcome} signaling",
    ],
}

# Keyword signals for detecting document-level relation
PREVENTION_SIGNALS = {"prevention", "preventive", "prophylaxis", "reduces risk",
                      "protective", "chemoprevention", "reduce incidence"}
TREATMENT_SIGNALS  = {"treatment", "therapy", "efficacy", "therapeutic", "cure",
                      "clinical trial", "randomized", "dose", "intervention"}
MECHANISM_SIGNALS  = {"pathway", "molecular", "in vitro", "mouse model", "cell line",
                      "kinase", "signaling", "in vivo", "mechanistic"}

# Relation types that are soft-penalised when they mismatch the claim
INCOMPATIBLE_RELATIONS = {
    "treatment":  {"prevention", "mechanism"},
    "prevention": {"treatment", "mechanism"},
    "causation":  {"mechanism"},
    "prognosis":  {"mechanism"},
}


def load_resources():
    """Load embedding model, ChromaDB collection, BM25 index, and abstracts."""
    if _cache:
        return (
            _cache["embedding_model"],
            _cache["collection"],
            _cache["bm25"],
            _cache["abstracts"]
        )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

    with open(f"{CHROMA_DB_PATH}/bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(f"{CHROMA_DB_PATH}/abstracts.pkl", "rb") as f:
        abstracts = pickle.load(f)

    _cache["embedding_model"] = embedding_model
    _cache["collection"] = collection
    _cache["bm25"] = bm25
    _cache["abstracts"] = abstracts

    return embedding_model, collection, bm25, abstracts


def dense_retrieve(query, embedding_model, collection, top_k=TOP_K, where=None):
    """
    Dense retrieval: embed query and find most similar abstracts in ChromaDB.
    Accepts an optional 'where' dict for metadata filtering.
    """
    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).tolist()

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["metadatas", "distances"]
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id": results["ids"][0][i],
            "doc_id": results["metadatas"][0][i].get("doc_id", ""),
            "title": results["metadatas"][0][i].get("title", ""),
            "text": results["metadatas"][0][i].get("text", ""),
            "source_abstract": results["metadatas"][0][i].get("source_abstract", ""),
            "score": 1 - results["distances"][0][i],
            "method": "dense"
        })

    return retrieved


def bm25_retrieve(query, bm25, abstracts, top_k=TOP_K):
    """
    BM25 sparse retrieval: keyword frequency matching.
    Preserves negation language -- contradicting papers score correctly.
    TREC BioGen 2025: BM25 achieved contradiction recall of 0.750.
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    retrieved = []
    for idx in top_indices:
        # BM25 operates on chunks (from abstracts.pkl which is now chunks)
        chunk = abstracts[idx]
        retrieved.append({
            "id": chunk.get("id", str(idx)),
            "doc_id": chunk.get("doc_id", ""),
            "title": chunk.get("title", ""),
            "text": chunk.get("text", chunk.get("text", "")),
            "source_abstract": chunk.get("source_abstract", ""),
            "score": float(scores[idx]),
            "method": "bm25"
        })

    return retrieved


def hybrid_retrieve(query, embedding_model, collection, bm25, abstracts, top_k=TOP_K, where=None):
    """
    Hybrid retrieval: reciprocal rank fusion of dense and BM25 results.
    Combines semantic understanding with keyword precision.
    Dense weight: 0.6, BM25 weight: 0.4 (configurable in config.py).
    """
    dense_results = dense_retrieve(query, embedding_model, collection, top_k=top_k * 2, where=where)
    
    # Optional metadata filtering is only fully supported by Chroma DB right now, 
    # but we can filter BM25 results post-retrieval or just let RRF handle it.
    # We fetch extra for BM25 to compensate.
    bm25_results = bm25_retrieve(query, bm25, abstracts, top_k=top_k * 3)
    if where and "species" in where:
        species_filter = where["species"]
        bm25_results = [r for r in bm25_results if abstracts[[a["id"] for a in abstracts].index(r["id"])].get("species") == species_filter]
        bm25_results = bm25_results[:top_k*2]

    # Reciprocal rank fusion
    rrf_scores = {}

    for rank, result in enumerate(dense_results):
        doc_id = result["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, {"score": 0, "data": result})
        rrf_scores[doc_id]["score"] += HYBRID_DENSE_WEIGHT * (1 / (rank + 1))

    for rank, result in enumerate(bm25_results):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {"score": 0, "data": result}
        rrf_scores[doc_id]["score"] += HYBRID_BM25_WEIGHT * (1 / (rank + 1))

    sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    retrieved = []
    for item in sorted_docs:
        result = item["data"]
        result["score"] = item["score"]
        result["method"] = "hybrid"
        retrieved.append(result)

    return retrieved


def apply_mmr(query, candidates, embedding_model, top_k=30, lambda_mult=0.5, paper_penalty=0.1):
    """
    Applies Maximal Marginal Relevance to diversify candidates before reranking.
    Diversifies based on chunk text embeddings and penalizes chunks from already-selected papers.
    """
    if not candidates:
        return []
        
    unique_papers_before = len(set(c.get("doc_id", c["id"]) for c in candidates))
        
    print(f"\n--- MMR Diversification ---")
    print(f"Original candidate pool: {len(candidates)} chunks across {unique_papers_before} papers")

    # Encode query and all candidates
    query_emb = embedding_model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    
    texts = [c["title"] + " " + c["text"] for c in candidates]
    doc_embs = embedding_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    # Cosine similarities
    sim_to_query = np.dot(doc_embs, query_emb)
    sim_matrix = np.dot(doc_embs, doc_embs.T)

    selected_indices = []
    unselected_indices = list(range(len(candidates)))
    selected_doc_ids = set()

    # First document is the one most similar to query
    first_idx = int(np.argmax(sim_to_query))
    selected_indices.append(first_idx)
    unselected_indices.remove(first_idx)
    selected_doc_ids.add(candidates[first_idx].get("doc_id", candidates[first_idx]["id"]))

    # Iteratively select the next best document
    while len(selected_indices) < top_k and unselected_indices:
        max_mmr = -np.inf
        best_idx = -1

        for i in unselected_indices:
            # Relevance to query
            rel = sim_to_query[i]
            
            # Max similarity to already selected docs (redundancy)
            redundancy = max([sim_matrix[i][j] for j in selected_indices])

            mmr_score = lambda_mult * rel - (1 - lambda_mult) * redundancy
            
            # Paper-level diversity penalty: prefer distinct papers when scores are similar
            doc_id = candidates[i].get("doc_id", candidates[i]["id"])
            if doc_id in selected_doc_ids:
                mmr_score -= paper_penalty

            if mmr_score > max_mmr:
                max_mmr = mmr_score
                best_idx = i

        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)
        selected_doc_ids.add(candidates[best_idx].get("doc_id", candidates[best_idx]["id"]))

    diversified = [candidates[i] for i in selected_indices]
    redundant_removed = len(candidates) - len(diversified)
    unique_papers_after = len(set(c.get("doc_id", c["id"]) for c in diversified))
    
    from collections import Counter
    paper_counts = Counter(c.get("doc_id", c["id"]) for c in diversified)
    
    print(f"Removed redundant chunks: {redundant_removed}")
    print(f"Final diversified set: {len(diversified)} chunks across {unique_papers_after} papers")
    print(f"Chunk-to-paper distribution: {dict(paper_counts.most_common(5))}...")
    print("---------------------------\n")

    return diversified

def _detect_doc_relation(text: str) -> str:
    """Heuristically detect the relation type expressed in a passage."""
    t = text.lower()
    if any(s in t for s in MECHANISM_SIGNALS):  return "mechanism"
    if any(s in t for s in PREVENTION_SIGNALS): return "prevention"
    if any(s in t for s in TREATMENT_SIGNALS):  return "treatment"
    return "unknown"


def _multi_query_retrieve(queries, embedding_model, collection, bm25, abstracts,
                          fetch_k, where):
    """
    Run multiple relation-typed queries and fuse via Reciprocal Rank Fusion.
    Earlier queries (more specific) receive higher importance weight.
    """
    all_scores = {}
    for q_idx, query in enumerate(queries):
        query_weight = 1.0 / (q_idx + 1)  # query 0 = most specific = highest weight

        dense_res = dense_retrieve(query, embedding_model, collection,
                                   top_k=fetch_k, where=where)
        bm25_res  = bm25_retrieve(query, bm25, abstracts, top_k=fetch_k)

        for rank, doc in enumerate(dense_res):
            key = doc.get("doc_id") or doc["id"]
            all_scores.setdefault(key, {"score": 0.0, "data": doc})
            all_scores[key]["score"] += HYBRID_DENSE_WEIGHT * query_weight * (1 / (rank + 1))

        for rank, doc in enumerate(bm25_res):
            key = doc.get("doc_id") or doc["id"]
            all_scores.setdefault(key, {"score": 0.0, "data": doc})
            all_scores[key]["score"] += HYBRID_BM25_WEIGHT * query_weight * (1 / (rank + 1))

    results = []
    for item in sorted(all_scores.values(), key=lambda x: x["score"], reverse=True)[:fetch_k]:
        doc = item["data"]
        doc["score"]  = item["score"]
        doc["method"] = "multi-query-hybrid"
        results.append(doc)
    return results


def retrieve(query, method=RETRIEVAL_METHOD, reformulated_query=None, top_k=TOP_K, where=None):
    """
    Main retrieval function. Calls the correct strategy based on config.
    method options: dense | bm25 | hybrid | queryreform

    For dense/hybrid: applies relation-aware multi-query retrieval.
    For all:          applies study-type weighting, paper-level deduplication,
                      relation mismatch penalty, and MMR diversification.
    """
    embedding_model, collection, bm25, abstracts = load_resources()

    intervention_terms = []
    outcome_terms      = []
    relation_type      = "association"
    claim_population   = "humans"
    queries            = [query]

    if method in ["dense", "hybrid"]:
        print("\n--- Query Decomposition (Relation-Aware) ---")
        components = decompose_claim(query)
        intervention_terms = components.get("intervention", [])
        outcome_terms      = components.get("outcome",      [])
        relation_type      = components.get("relation_type", "association")
        claim_population   = components.get("population",    "humans")
        relation_verb      = components.get("relation",      "")

        print(f"Intervention   : {intervention_terms}")
        print(f"Outcome        : {outcome_terms}")
        print(f"Relation type  : {relation_type} ({relation_verb!r})")
        print(f"Population     : {claim_population}")

        # Hard species filter for human-specific claims
        if claim_population == "humans" and where is None:
            try:
                where = {"species": "human"}
                print("Applied hard filter: where={species: human}")
            except Exception:
                where = None  # ChromaDB may not have enough human docs

        # Build relation-typed queries
        iv = intervention_terms[0] if intervention_terms else query
        oc = outcome_terms[0]      if outcome_terms      else ""
        templates = RELATION_QUERY_TEMPLATES.get(relation_type, ["{intervention} {outcome}"])
        queries   = [t.format(intervention=iv, outcome=oc) for t in templates]
        print(f"Relation queries: {queries}\n")

    elif method == "queryreform" and reformulated_query:
        queries = [reformulated_query]

    # We fetch 2x candidates so study-type weighting + deduplication have a good pool
    fetch_k = top_k * 2

    if method in ["dense", "hybrid"]:
        candidates = _multi_query_retrieve(
            queries, embedding_model, collection, bm25, abstracts, fetch_k, where)
    elif method == "bm25":
        candidates = bm25_retrieve(queries[0], bm25, abstracts, top_k=fetch_k)
    elif method == "queryreform":
        candidates = dense_retrieve(queries[0], embedding_model, collection,
                                    top_k=fetch_k, where=where)
    else:
        raise ValueError(f"Unknown retrieval method: {method}. Choose: dense | bm25 | hybrid | queryreform")

    print(f"Multi-query retrieval returned {len(candidates)} candidates.")

    # Boost chunks containing BOTH intervention and outcome terms
    if intervention_terms and outcome_terms:
        boosted_count = 0
        for c in candidates:
            text_lower = (c.get("title", "") + " " + c.get("text", "")).lower()
            has_intervention = any(i.lower() in text_lower for i in intervention_terms)
            has_outcome      = any(o.lower() in text_lower for o in outcome_terms)
            if has_intervention and has_outcome:
                c["score"] *= 1.5
                boosted_count += 1
        candidates.sort(key=lambda x: x["score"], reverse=True)
        print(f"--- Boosting ---")
        print(f"Boosted {boosted_count} chunks containing both intervention and outcome terms.\n")

    # Relation mismatch penalty — soft-penalise documents with incompatible relation type
    incompatible = INCOMPATIBLE_RELATIONS.get(relation_type, set())
    if incompatible:
        penalised = 0
        for c in candidates:
            doc_rel = _detect_doc_relation(c.get("text", ""))
            if doc_rel in incompatible:
                c["score"]             *= 0.6
                c["relation_mismatch"] = True
                penalised += 1
        if penalised:
            print(f"--- Relation Mismatch Penalty ---")
            print(f"Penalised {penalised} candidates with incompatible relation type (claim={relation_type})")
            print(f"---------------------------------\n")
            candidates.sort(key=lambda x: x["score"], reverse=True)

    # Study-type weighting: boost evidence from higher-quality study designs
    STUDY_WEIGHT = {
        "meta-analysis": 1.5,
        "rct":           1.3,
        "review":        1.0,
        "case report":   0.9,
        "unknown":       0.8,
    }
    for c in candidates:
        study_type = c.get("study_type", "unknown").lower()
        multiplier = STUDY_WEIGHT.get(study_type, 0.8)
        if multiplier != 1.0:
            c["score"] *= multiplier

    # Paper-level MaxPool deduplication — keep only the best chunk per paper
    paper_best = {}
    for c in candidates:
        doc_id = c.get("doc_id") or c.get("id", "")
        if doc_id not in paper_best or c["score"] > paper_best[doc_id]["score"]:
            paper_best[doc_id] = c

    deduped = sorted(paper_best.values(), key=lambda x: x["score"], reverse=True)
    print(f"--- Paper-Level Deduplication ---")
    print(f"Before: {len(candidates)} chunks | After: {len(deduped)} unique papers")
    print(f"---------------------------------\n")

    # Apply MMR to reduce to the requested top_k before returning to the reranker
    return apply_mmr(query, deduped, embedding_model, top_k=top_k)




if __name__ == "__main__":
    test_claim = "Vitamin D supplementation improves depression symptoms"
    print(f"Test claim: {test_claim}")
    print(f"Retrieval method: {RETRIEVAL_METHOD}\n")

    results = retrieve(test_claim)

    for i, r in enumerate(results):
        print(f"[{i+1}] {r['title']}")
        print(f"     Score: {r['score']:.4f} | Method: {r['method']}")
        print(f"     {r['text'][:150]}...")
        print()
