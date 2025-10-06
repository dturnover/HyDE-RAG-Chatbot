# config.py
import os
from pathlib import Path

# --- API and Model Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# --- RAG_DATA_PATH is now defined in rag.py ---

# --- Chaplain Logic Keywords ---
ASK_WORDS = {"verse", "scripture", "psalm", "quote", "passage", "bible"}
DISTRESS_KEYWORDS = {"scared", "anxious", "worried", "nervous", "afraid", "stress"}
FAITH_KEYWORDS = {
    "catholic": "bible_nrsv",
    "christian": "bible_asv",
    "protestant": "bible_asv",
    "jewish": "tanakh",
    "muslim": "quran",
    "islam": "quran",
    "hindu": "gita",
}