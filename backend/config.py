# config.py
import os
from pathlib import Path

# --- API and Model Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# --- RAG Index Configuration ---
def get_rag_data_path() -> Path:
    """Finds the directory containing the RAG index files."""
    here = Path(__file__).parent.resolve()
    # Add potential directories to search for 'indexes'
    search_paths = [
        Path(os.getenv("RAG_DIR", "")),
        here / "indexes",
        here.parent / "indexes",
        Path("/opt/render/project/src/indexes") # For specific deployment environments
    ]
    for p in search_paths:
        if p and p.is_dir():
            return p
    return here # Default fallback

RAG_DATA_PATH = get_rag_data_path()
BIG_CORPORA = {"bible_nrsv", "tanakh", "bible_asv"}

# --- Chaplain Logic Keywords ---
ASK_WORDS = {
    "verse", "scripture", "psalm", "quote", "passage", "ayah", "surah",
    "quran", "bible", "tanakh", "gita", "dhammapada"
}
DISTRESS_KEYWORDS = {
    "scared", "anxious", "worried", "hurt", "down", "lost", "depressed",
    "angry", "grief", "lonely", "alone", "doubt", "stress", "nervous", "afraid"
}
CRISIS_KEYWORDS = {
    "panic", "suicide", "kill myself", "hopeless", "end it", "emergency", "self-harm"
}
FAITH_KEYWORDS = {
    "catholic": "bible_nrsv", "orthodox": "bible_nrsv", "protestant": "bible_asv",
    "evangelical": "bible_asv", "christian": "bible_asv", "jewish": "tanakh",
    "jew": "tanakh", "hebrew": "tanakh", "muslim": "quran", "islam": "quran",
    "quran": "quran", "koran": "quran", "hindu": "gita", "bhagavad": "gita",
    "gita": "gita", "buddhist": "dhammapada", "buddhism": "dhammapada",
    "dhammapada": "dhammapada",
}