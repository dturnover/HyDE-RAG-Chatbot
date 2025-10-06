# rag.py
import re, json, math
from typing import Dict, List, Any, Optional
from pathlib import Path
from openai import OpenAI
import config # Still needed for FAITH_KEYWORDS

client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None

# ★★★ THE FINAL FIX IS HERE ★★★
# Define the path directly in the file that uses it to avoid import issues.
RAG_DATA_PATH = Path("/opt/render/project/src/indexes")

def embed_query(text: str) -> Optional[List[float]]:
    # ... function is unchanged
    if not client: return None
    try:
        response = client.embeddings.create(model=config.EMBED_MODEL, input=text)
        return response.data[0].embedding
    except Exception: return None

def tokenize(s: str) -> set[str]:
    # ... function is unchanged
    return set(re.findall(r"\w+", s.lower()))

def cos_sim(a: List[float], b: List[float]) -> float:
    # ... function is unchanged
    if not a or not b: return 0.0
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

def hybrid_search(query: str, corpus_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
    file_name = f"{corpus_name}_embed.jsonl"
    # Use the RAG_DATA_PATH defined directly in this file
    path = RAG_DATA_PATH / file_name

    if not path.exists():
        print(f"[DEBUG hybrid_search] Corpus path not found at: {path}. Exiting.")
        return []

    # ... the rest of the search logic is unchanged ...
    q_tokens = tokenize(query)
    q_emb = embed_query(query)
    if not q_emb: return []

    alpha = 0.2
    scored_docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                text = row.get("text", "")
                embedding = row.get("embedding")
                if not text or not isinstance(embedding, list): continue

                lexical_score = jaccard(q_tokens, tokenize(text))
                vector_score = cos_sim(q_emb, embedding)
                final_score = (alpha * lexical_score) + ((1 - alpha) * vector_score)
                
                if final_score > 0.22:
                    scored_docs.append((final_score, row))
            except (json.JSONDecodeError, TypeError):
                continue
    
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:top_k]]