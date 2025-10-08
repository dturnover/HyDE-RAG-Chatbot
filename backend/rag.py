# rag.py
import re, json, math
from typing import Dict, List, Any, Optional
from pathlib import Path
from openai import OpenAI
import config

client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
RAG_DATA_PATH = Path("/opt/render/project/src/indexes")

# ★★★ EXPANDED SYNONYM MAP ★★★
SYNONYM_MAP = {
    # Fear / Anxiety
    "nervous": {"fear", "afraid", "anxious", "courage", "strength", "worry", "peace"},
    "anxious": {"fear", "afraid", "anxious", "courage", "strength", "worry", "peace"},
    "scared":  {"fear", "afraid", "anxious", "courage", "strength", "worry", "peace"},
    "afraid":  {"fear", "afraid", "anxious", "courage", "strength", "worry", "peace"},
    "worried": {"fear", "afraid", "anxious", "courage", "strength", "worry", "peace"},
    "stress":  {"burden", "peace", "rest", "anxious", "worry", "strength"},
    # Sadness / Loss
    "sad":       {"sorrow", "mourn", "weep", "comfort", "joy", "despair", "broken", "spirit"},
    "grief":     {"sorrow", "mourn", "weep", "comfort", "death", "loss"},
    "hurting":   {"heal", "pain", "suffer", "broken", "comfort", "refuge"},
    "lost":      {"found", "guide", "way", "path", "seek", "shepherd"},
    "lonely":    {"alone", "comfort", "friend", "presence", "god", "lord"},
    "heartbroken": {"heal", "broken", "heart", "comfort", "sorrow", "love"},
    "depressed": {"despair", "hope", "lift", "spirit", "light", "darkness"},
    # Anger / Frustration
    "angry":       {"anger", "wrath", "rage", "forgive", "patience", "peace", "justice"},
    "frustrated":  {"patience", "peace", "anger", "rest", "striving"},
    "betrayed":    {"trust", "friend", "forgive", "love", "justice", "enemy"},
    "resentful":   {"bitter", "forgive", "heart", "love", "peace"},
    "bitter":      {"bitter", "forgive", "heart", "love", "peace"},
    # Doubt / Uncertainty
    "doubt":     {"faith", "believe", "trust", "wisdom", "understanding", "seek"},
    "confused":  {"wisdom", "guidance", "understanding", "light", "path", "clarity"},
    "uncertain": {"faith", "trust", "hope", "guide", "future", "path"},
    "conflicted": {"peace", "heart", "mind", "wisdom", "guide"},
    # Weakness / Failure
    "weak":        {"strength", "power", "strong", "lift", "spirit", "grace"},
    "failure":     {"fail", "fall", "rise", "grace", "forgive", "mercy", "redeem"},
    "overwhelmed": {"burden", "rest", "peace", "strength", "help", "refuge"},
    "guilt":       {"sin", "forgive", "mercy", "cleanse", "grace", "redeem"},
    "shame":       {"sin", "forgive", "mercy", "honor", "grace", "glory"},
    "stuck":       {"free", "deliver", "path", "way", "hope"},
}


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
    file_name = f"{corpus_name}.jsonl"
    path = RAG_DATA_PATH / file_name
    if not path.exists(): return []

    q_tokens = tokenize(query)
    q_emb = embed_query(query)
    if not q_emb: return []

    search_tokens = set(q_tokens)
    for token in q_tokens:
        if token in SYNONYM_MAP:
            search_tokens.update(SYNONYM_MAP[token])

    candidate_rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if any(token in line.lower() for token in search_tokens):
                try:
                    candidate_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    if not candidate_rows: return []

    alpha = 0.05
    scored_docs = []
    for row in candidate_rows:
        text = row.get("text", "")
        embedding = row.get("embedding")
        if not text or not isinstance(embedding, list): continue

        vector_score = cos_sim(q_emb, embedding)
        lexical_score = 1 if any(token in text.lower() for token in q_tokens) else 0
        final_score = (alpha * lexical_score) + ((1 - alpha) * vector_score)
        
        if final_score > 0.22:
            scored_docs.append((final_score, row))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:top_k]]