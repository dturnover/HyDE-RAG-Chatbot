# rag.py
import re, json, math
from typing import Dict, List, Any, Optional
from pathlib import Path
from openai import OpenAI
import config

client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
RAG_DATA_PATH = Path("/opt/render/project/src/indexes")

def embed_query(text: str) -> Optional[List[float]]:
    # ... (unchanged)
    if not client: return None
    try:
        response = client.embeddings.create(model=config.EMBED_MODEL, input=text)
        return response.data[0].embedding
    except Exception: return None

def tokenize(s: str) -> set[str]:
    # ... (unchanged)
    return set(re.findall(r"\w+", s.lower()))

def cos_sim(a: List[float], b: List[float]) -> float:
    # ... (unchanged)
    if not a or not b: return 0.0
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

def hybrid_search(query: str, corpus_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
    file_name = f"{corpus_name}.jsonl"
    path = RAG_DATA_PATH / file_name
    if not path.exists(): return []

    q_tokens = tokenize(query)
    q_emb = embed_query(query)
    if not q_emb: return []

    # ★★★ THE FINAL FIX: A TRULY LOW-MEMORY FIRST PASS ★★★
    candidate_rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            # Check for keywords without creating a new copy of the line in memory.
            # This is more complex but necessary for large files in low-RAM environments.
            found_match = False
            for token in q_tokens:
                if token in line: # Case-sensitive check first for speed
                    found_match = True
                    break
                # Fallback to slower, case-insensitive check if needed
                if re.search(r'\b' + re.escape(token) + r'\b', line, re.IGNORECASE):
                    found_match = True
                    break
            
            if found_match:
                try:
                    candidate_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not candidate_rows: return []

    # --- Pass 2: Vector Search on Candidates ---
    scored_docs = []
    for row in candidate_rows:
        embedding = row.get("embedding")
        if isinstance(embedding, list):
            vector_score = cos_sim(q_emb, embedding)
            if vector_score > 0.3:
                scored_docs.append((vector_score, row))
    
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:top_k]]