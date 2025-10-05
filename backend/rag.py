# rag.py
import re, json, math, heapq
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from openai import OpenAI
import config

# --- OpenAI Client Initialization ---
client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None

# --- Corpus Discovery ---
CORPUS_FILES: Dict[str, Path] = {
    key: config.RAG_DATA_PATH / f"{key}_embed.jsonl"
    for key in config.FAITH_KEYWORDS.values()
}

def get_available_corpora() -> Dict[str, Path]:
    """Returns a dictionary of corpus names to paths that actually exist."""
    return {k: v for k, v in CORPUS_FILES.items() if v.exists() and v.stat().st_size > 0}

AVAILABLE_CORPORA = get_available_corpora()

# --- Math & Text Utilities for Hybrid Search ---

def tokenize(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))

def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def cos_sim(a: List[float], b: List[float]) -> float:
    if not a or not b: return 0.0
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

@dataclass
class Stat:
    """Welford's algorithm for stable, one-pass mean/std calculation."""
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def push(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        if self.n < 2: return 1.0 # Avoid division by zero
        return math.sqrt(self.m2 / self.n)

# --- Core RAG Logic ---

def embed_query(text: str) -> Optional[List[float]]:
    if not client: return None
    try:
        response = client.embeddings.create(model=config.EMBED_MODEL, input=text)
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding failed: {e}")
        return None

def hybrid_search(query: str, corpus_name: str, top_k: int = 6) -> List[Dict[str, Any]]:
    path = AVAILABLE_CORPORA.get(corpus_name)
    if not path: return []

    q_tokens = tokenize(query)
    q_emb = embed_query(query)

    lex_heap, vec_heap = [], []
    lex_stat, vec_stat = Stat(), Stat()
    candidates: Dict[int, Dict] = {}
    lex_scores, vec_scores = {}, {}

    # Single pass to score and gather stats
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                row = json.loads(line)
                candidates[i] = row
                
                # Lexical Score
                text_tokens = tokenize(row.get("text", ""))
                js = jaccard(q_tokens, text_tokens)
                lex_scores[i] = js
                lex_stat.push(js)
                heapq.heappush(lex_heap, (js, i))
                if len(lex_heap) > 50: heapq.heappop(lex_heap)

                # Vector Score
                if q_emb and "embedding" in row:
                    cs = cos_sim(q_emb, row["embedding"])
                    vec_scores[i] = cs
                    vec_stat.push(cs)
                    heapq.heappush(vec_heap, (cs, i))
                    if len(vec_heap) > 50: heapq.heappop(vec_heap)

            except json.JSONDecodeError:
                continue
    
    # Consolidate top candidates from both searches
    top_indices = {idx for _, idx in lex_heap} | {idx for _, idx in vec_heap}
    if not top_indices: return []

    # Normalize scores and blend
    mean_l, std_l = lex_stat.mean, lex_stat.std
    mean_v, std_v = vec_stat.mean, vec_stat.std
    
    # ★★★ THE FIX IS HERE ★★★
    alpha = 0.2  # 80% semantic, 20% lexical
    
    blended_scores = []
    for idx in top_indices:
        lz = (lex_scores.get(idx, 0) - mean_l) / std_l
        vz = (vec_scores.get(idx, 0) - mean_v) / std_v
        final_score = (alpha * lz) + ((1 - alpha) * vz)
        blended_scores.append((final_score, candidates[idx]))

    # Sort by blended score and return top_k
    blended_scores.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in blended_scores[:top_k]]