# rag.py — ultra-light vector search over JSONL (embeddings + text)
import os, json
from typing import List, Dict, Optional, Tuple
import numpy as np

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

class VectorIndex:
    def __init__(self, name: str):
        self.name = name
        self.meta: List[Dict] = []
        self.mat: Optional[np.ndarray] = None  # shape (N, D)

    def load_jsonl(self, path: str) -> int:
        metas, vecs = [], []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                emb = obj.get("embedding")
                text = obj.get("text")
                if emb is None or text is None:
                    continue
                metas.append({
                    "id": obj.get("id", ""),
                    "text": text,
                    "ref": obj.get("ref") or obj.get("book") or "",
                    "source": obj.get("source", self.name),
                })
                vecs.append(emb)
        if not vecs:
            self.meta, self.mat = [], None
            return 0
        self.meta = metas
        self.mat = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(self.mat, axis=1, keepdims=True) + 1e-8
        self.mat = self.mat / norms  # pre-normalize for cosine
        return len(self.meta)

    def search(self, q_vec: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        if self.mat is None or self.mat.size == 0: return []
        qn = q_vec / (np.linalg.norm(q_vec) + 1e-8)
        sims = self.mat @ qn  # cosine
        k = min(top_k, sims.size) if sims.size else 0
        if k == 0: return []
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[int(i)])) for i in idx]

class RAGStore:
    def __init__(self):
        self.idxs: Dict[str, VectorIndex] = {}

    def load_all(self, base_dir: str):
        for trad in ["bible", "quran", "talmud"]:
            p1 = os.path.join(base_dir, f"{trad}.jsonl")
            p2 = os.path.join(os.path.dirname(__file__), f"{trad}.jsonl")
            path = p1 if os.path.exists(p1) else p2 if os.path.exists(p2) else None
            if not path:
                continue
            vi = VectorIndex(trad)
            n = vi.load_jsonl(path)
            if n > 0:
                self.idxs[trad] = vi

    def traditions(self) -> List[str]:
        return list(self.idxs.keys())

    def embed_query(self, client, text: str) -> Optional[np.ndarray]:
        if client is None:
            # DEV fallback: deterministic pseudo-vector
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            return rng.standard_normal(1536).astype(np.float32)
        emb = client.embeddings.create(model=EMBED_MODEL, input=text)
        return np.asarray(emb.data[0].embedding, dtype=np.float32)

    def pick_trads(self, hint: str, default_all: bool = True) -> List[str]:
        m = hint.lower()
        if "bible" in m or "psalm" in m or "christ" in m: return ["bible"] if "bible" in self.idxs else []
        if "qur" in m or "islam" in m or "surah" in m or "muslim" in m: return ["quran"] if "quran" in self.idxs else []
        if "talmud" in m or "rabbi" in m or "tractate" in m or "jew" in m: return ["talmud"] if "talmud" in self.idxs else []
        return self.traditions() if default_all else []

    def retrieve(self, client, query: str, hint: str = "", top_k_each: int = 4, limit: int = 6) -> List[Dict]:
        qv = self.embed_query(client, query)
        if qv is None: return []
        trads = self.pick_trads(hint, default_all=True)
        results: List[Tuple[str, int, float]] = []
        for t in trads:
            idx = self.idxs.get(t)
            if not idx: continue
            for i, s in idx.search(qv, top_k=top_k_each):
                results.append((t, i, s))
        results.sort(key=lambda x: -x[2])
        out: List[Dict] = []
        seen = set()
        for t, i, s in results:
            vi = self.idxs[t]
            meta = vi.meta[i]
            key = (t, meta["id"])
            if key in seen: continue
            seen.add(key)
            out.append({
                "trad": t,
                "id": meta["id"],
                "ref": meta.get("ref", ""),
                "text": meta["text"],
                "score": s,
            })
            if len(out) >= limit: break
        return out

STORE = RAGStore()

def init_store():
    base = os.getenv("RAG_DIR", os.path.join(os.path.dirname(__file__), "indexes"))
    STORE.load_all(base_dir=base)
