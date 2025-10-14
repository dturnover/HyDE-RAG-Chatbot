# config.py
import os
from pathlib import Path

# --- API and Model Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# --- Chaplain Logic Keywords ---
ASK_WORDS = {"verse", "scripture", "psalm", "quote", "passage", "bible"}

DISTRESS_KEYWORDS = {
    # Fear / Anxiety
    "scared", "anxious", "worried", "nervous", "afraid", "stress", "panic", "terrified",
    # Sadness / Loss
    "sad", "grief", "hurting", "lost", "lonely", "heartbroken", "depressed", "discouraged", "hopeless",
    # Anger / Frustration
    "angry", "frustrated", "betrayed", "resentful", "bitter", "jealous", "furious",
    # Doubt / Uncertainty
    "doubt", "confused", "uncertain", "conflicted", "insecure",
    # Weakness / Failure
    "weak", "failure", "overwhelmed", "guilt", "shame", "stuck", "tempted", "sin"
}

# ★★★ CORRECT BIBLE VERSION MAPPING ★★★
FAITH_KEYWORDS = {
    # Catholic uses the NRSV translation
    "catholic": "bible_nrsv",
    
    # Other Christian denominations use the ASV translation
    "christian": "bible_asv",
    "protestant": "bible_asv",
    "baptist": "bible_asv",
    "methodist": "bible_asv",
    "lutheran": "bible_asv",

    # Other Religions
    "jewish": "tanakh",
    "judaism": "tanakh",
    "muslim": "quran",
    "islam": "quran",
    "hindu": "gita",
}