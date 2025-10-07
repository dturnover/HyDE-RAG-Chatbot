# rag.py
import re, json, math
from typing import Dict, List, Any, Optional
from pathlib import Path
from openai import OpenAI
import config

client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
RAG_DATA_PATH = Path("/opt/render/project/src/indexes")
CORPUS_CACHE: Dict[str, List[Dict]] = {}

def load_corpus(corpus_name: str) -> List[Dict[str, Any]]:
    if corpus_name in CORPUS_CACHE:
        return CORPUS_CACHE[corpus_name]

    file_name = f"{corpus_name}.jsonl"
    path = RAG_DATA_PATH / file_name
    if not path.exists(): return []

    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try: docs.append(json.loads(line))
            except json.JSONDecodeError: continue
    
    CORPUS_CACHE[corpus_name] = docs
    return docs

def embed_query(text: str) -> Optional[List[float]]:
    if not client: return None
    try:
        response = client.embeddings.create(model=config.EMBED_MODEL, input=text)
        return response.data[0].embedding
    except Exception: return None

def tokenize(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))

def cos_sim(a: List[float], b: List[float]) -> float:
    if not a or not b: return 0.0
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

def hybrid_search(query: str, corpus_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
    docs = load_corpus(corpus_name)
    if not docs: return []

    q_tokens = tokenize(query)
    q_emb = embed_query(query)
    if not q_emb: return []

    alpha = 0.05
    scored_docs = []
    for row in docs:
        text = row.get("text", "")
        embedding = row.get("embedding")
        if not text or not isinstance(embedding, list): continue

        jaccard_score = 0.0 # Lexical score is not needed for this implementation
        vector_score = cos_sim(q_emb, embedding)
        final_score = (alpha * jaccard_score) + ((1 - alpha) * vector_score)
        
        if final_score > 0.22:
            scored_docs.append((final_score, row))
    
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # ★★★ DEBUG STATEMENTS ADDED HERE ★★★
    print("\n--- [DEBUG] Hybrid Search Diagnostics ---")
    print(f"[DEBUG] Query: '{query}'")
    print("\n[DEBUG] Top 10 potential matches (before final selection):")
    for i, (score, doc) in enumerate(scored_docs[:10]):
        ref = doc.get('ref') or f"{doc.get('book', '')} {doc.get('chapter', '')}:{doc.get('verse', '')}".strip()
        print(f"[DEBUG]   {i+1}. Score: {score:.4f} | Ref: {ref} | Text: '{doc.get('text', '')[:70]}...'")
    print("--- [DEBUG] End Diagnostics ---\n")
    
    return [doc for score, doc in scored_docs[:top_k]]