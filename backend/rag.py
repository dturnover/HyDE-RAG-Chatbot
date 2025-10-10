# rag.py
import re, json, math, heapq
from typing import Dict, List, Any, Optional
from pathlib import Path
from openai import OpenAI
import config

client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
RAG_DATA_PATH = Path("/opt/render/project/src/indexes")

def embed_query(text: str) -> Optional[List[float]]:
    if not client: return None
    try:
        response = client.embeddings.create(model=config.EMBED_MODEL, input=text)
        return response.data[0].embedding
    except Exception as e:
        print(f"[DEBUG rag.py] ERROR during embedding: {e}")
        return None

def tokenize(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))

def cos_sim(a: List[float], b: List[float]) -> float:
    if not a or not b: return 0.0
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

def hybrid_search(query: str, corpus_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
    print("\n--- [DEBUG rag.py] Entering hybrid_search ---")
    print(f"[DEBUG rag.py] Received query for search: '{query}'")

    file_name = f"{corpus_name}.jsonl"
    path = RAG_DATA_PATH / file_name
    if not path.exists():
        print(f"[DEBUG rag.py] ERROR: Corpus path not found at: {path}")
        return []

    q_tokens = tokenize(query)
    q_emb = embed_query(query)
    if not q_emb:
        print("[DEBUG rag.py] ERROR: Failed to create embedding for query.")
        return []

    print(f"[DEBUG rag.py] Tokens for Pass 1: {q_tokens}")

    # ★★★ THE FINAL FIX IS HERE: A FAST AND MEMORY-SAFE FIRST PASS ★★★
    candidate_rows = []
    # Build a single, efficient regex to find any of the tokens case-insensitively.
    # We only search for tokens longer than 2 characters to avoid noise (e.g., 'a', 'is', 'to').
    searchable_tokens = [t for t in q_tokens if len(t) > 2]
    if searchable_tokens:
        pattern = re.compile(r'\b(' + '|'.join(re.escape(t) for t in searchable_tokens) + r')\b', re.IGNORECASE)
        print(f"[DEBUG rag.py] Compiled regex pattern for Pass 1: {pattern.pattern}")

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if pattern.search(line):
                    try:
                        candidate_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    print(f"[DEBUG rag.py] Pass 1 (regex scan) found {len(candidate_rows)} candidates.")
    if not candidate_rows:
        print("--- [DEBUG rag.py] Leaving hybrid_search (no candidates found) ---\n")
        return []

    # --- Pass 2: Vector Search on Candidates ---
    scored_docs = []
    for row in candidate_rows:
        embedding = row.get("embedding")
        if isinstance(embedding, list):
            vector_score = cos_sim(q_emb, embedding)
            if vector_score > 0.3:
                scored_docs.append((vector_score, row))
    
    print(f"[DEBUG rag.py] Pass 2 (vector search) found {len(scored_docs)} matches above threshold.")
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    final_results = [doc for score, doc in scored_docs[:top_k]]
    print(f"[DEBUG rag.py] Returning top {len(final_results)} results.")
    print("--- [DEBUG rag.py] Leaving hybrid_search ---\n")
    return final_results