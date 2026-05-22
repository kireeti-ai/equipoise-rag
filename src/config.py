import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# --- Models ---
GROQ_VERDICT_MODEL = "meta-llama/llama-3.3-70b-instruct"
GROQ_REFORMULATOR_MODEL = "meta-llama/llama-3.3-70b-instruct"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "ncbi/MedCPT-Cross-Encoder"

# --- Retrieval Settings ---
RETRIEVAL_METHOD = "hybrid"     # options: dense | bm25 | hybrid | queryreform
RETRIEVAL_CANDIDATE_K = 30     # documents fetched before reranking
TOP_K = 5                      # final documents kept after reranking (INV-02: 3, 5, or 10)
HYBRID_DENSE_WEIGHT = 0.55
HYBRID_BM25_WEIGHT = 0.45

# --- Prompt Variant ---
PROMPT_VARIANT = "structured"  # INV-03: options: neutral | biased | structured

# --- ChromaDB ---
CHROMA_DB_PATH = "chroma_db"
CHROMA_COLLECTION_NAME = "scifact_abstracts"

# --- Local Data Paths ---
CORPUS_PATH = "data/scifact/data/corpus.jsonl"
CLAIMS_TRAIN_PATH = "data/scifact/data/claims_train.jsonl"
CLAIMS_DEV_PATH = "data/scifact/data/claims_dev.jsonl"

# --- Evaluation ---
VALIDATION_SPLIT_SIZE = 300

# --- Paths ---
RESULTS_DIR = "results"
