# rag.py
import re, json, math
from typing import Dict, List, Any, Optional
from pathlib import Path
from openai import OpenAI
import config

client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
RAG_DATA_PATH = Path("/opt/render/project/src/indexes")

# ★★★ FIX 1: CACHING FOR SPEED ★★★
# This dictionary will hold corpora once they are loaded into memory.
CORPUS_CACHE: Dict[str, List[Dict]] = {}

def load_corpus(corpus_name: str) -> List[Dict[str, Any]]:
    """Loads a corpus from disk into memory and caches it."""
    if corpus_name in CORPUS_CACHE:
        return CORPUS_CACHE[corpus_name]

    file_name = f"{corpus_name}.jsonl"
    path = RAG_DATA_PATH / file_name
    if not path.exists():
        return []

    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    CORPUS_CACHE[corpus_name] = docs
    return docs

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
    docs = load_corpus(corpus_name)
    if not docs:
        return []

    q_tokens = tokenize(query)
    q_emb = embed_query(query)
    if not q_emb: return []

    # ★★★ FIX 3: BETTER QUOTE QUALITY ★★★
    # Change alpha to make the search 95% semantic.
    alpha = 0.05

    scored_docs = []
    for row in docs:
        text = row.get("text", "")
        embedding = row.get("embedding")
        if not text or not isinstance(embedding, list): continue

        jaccard_score = jaccard(q_tokens, tokenize(text))
        vector_score = cos_sim(q_emb, embedding)
        final_score = (alpha * jaccard_score) + ((1 - alpha) * vector_score)
        
        if final_score > 0.22:
            scored_docs.append((final_score, row))
    
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:top_k]]