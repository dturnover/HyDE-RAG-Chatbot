# config.py
import os
from pathlib import Path

# --- API and Model Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# --- RAG Index Configuration ---
def get_rag_data_path() -> Path:
    here = Path(__file__).parent.resolve()
    search_paths = [
        Path(os.getenv("RAG_DIR", "")),
        here / "indexes",
        here.parent / "indexes",
        Path("/opt/render/project/src/indexes")
    ]
    for p in search_paths:
        if p and p.is_dir():
            return p
    return here

RAG_DATA_PATH = get_rag_data_path()

# --- Chaplain Logic Keywords ---
ASK_WORDS = {"verse", "scripture", "psalm", "quote", "passage", "bible"}
DISTRESS_KEYWORDS = {"scared", "anxious", "worried", "nervous", "afraid", "stress"}
FAITH_KEYWORDS = {
    "catholic": "bible_nrsv", "christian": "bible_asv", "protestant": "bible_asv",
    "jewish": "tanakh", "muslim": "quran", "islam": "quran", "hindu": "gita"
}