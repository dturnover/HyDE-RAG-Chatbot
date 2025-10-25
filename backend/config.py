# config.py
# Contains API keys, model configs, and expanded keyword lists for RAG.
import os
from pathlib import Path
from dotenv import load_dotenv # Added load_dotenv for consistency

# Load environment variables from .env file (if you use one)
load_dotenv()

# --- API and Model Configuration ---
# (Using your os.getenv structure)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Check if the key is loaded
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY environment variable not found.")
    # You might want to raise an error here if running in production
    # raise ValueError("OPENAI_API_KEY environment variable not set.")

# --- Faith Mapping ---
# (Expanded to include all faiths)
FAITH_KEYWORDS = {
    # Christian / Catholic
    "christian": "bible_nrsv", # Default Christian to NRSV as Catholic is specified
    "catholic": "bible_nrsv",
    "protestant": "bible_asv",
    "baptist": "bible_asv",
    "methodist": "bible_asv",
    "lutheran": "bible_asv",
    "anglican": "bible_nrsv",
    "evangelical": "bible_asv",
    "orthodox": "bible_nrsv", # Eastern Orthodox
    "jesus": "bible_nrsv",
    "bible": "bible_nrsv", # Default Bible to NRSV
    "asv": "bible_asv", # Specific version
    "nrsv": "bible_nrsv", # Specific version
    "king james": "bible_asv", # Assuming ASV is closer if KJV requested

    # Jewish
    "jewish": "tanakh",
    "judaism": "tanakh",
    "hebrew": "tanakh",
    "tanakh": "tanakh",
    "torah": "tanakh",

    # Muslim
    "muslim": "quran",
    "islam": "quran",
    "quran": "quran",
    "koran": "quran",

    # Hindu
    "hindu": "gita",
    "gita": "gita",
    "bhagavad gita": "gita",
    "krishna": "gita",

    # Buddhist
    "buddhist": "dhammapada",
    "buddhism": "dhammapada",
    "dhammapada": "dhammapada",
}

# --- RAG Trigger Keywords ---

# Keywords indicating a state of distress
# (Expanded to be "bulletproof")
DISTRESS_KEYWORDS = {
    # Fear / Anxiety
    "afraid", "alone", "anxiety", "anxious", "apprehensive", "dread",
    "fear", "frightened", "freaking out", "nervous", "on edge",
    "panicked", "panicking", "scared", "stress", "stressed", "tense",
    "terrified", "uneasy", "worried",

    # Sadness / Depression
    "blue", "broken", "heartbroken", "depressed", "depression", "despair",
    "discouraged", "down", "grief", "grieving", "hopeless", "hurting",
    "low", "lonely", "lost", "miserable", "mourning", "sad", "sorrow",
    "unhappy",

    # Anger
    "angry", "annoyed", "betrayed", "bitter", "frustrated", "furious",
    "irritated", "jealous", "mad", "pissed", "rage", "resentful", "upset",

    # General Struggle / Pain
    "battling", "broken", "burnt out", "conflicted", "confused", "crushed",
    "defeated", "difficult", "drained", "empty", "exhausted", "failure",
    "failing", "guilt", "hard time", "insecure", "overwhelmed", "pain",
    "shame", "sin", "struggling", "stuck", "suffering", "tempted",
    "terrible", "tired", "trouble", "weak", "worn out"
}

# Keywords explicitly or implicitly asking for help/scripture
# (Expanded to be "bulletproof")
ASK_WORDS = {
    "advise", "advice",
    "bible",
    "comfort", "courage",
    "dhammapada",
    "gita", "guidance", "guide",
    "help", "hope",
    "koran",
    "passage", "peace", "pray", "prayer",
    "quran", "quote",
    "scripture", "strength", "support",
    "tanakh", "text", "torah",
    "verse",
    "wisdom"
}

# --- Escalation Keywords ---
# (Added - This is CRITICAL for logic.py to run)
CRISIS_KEYWORDS = {
    "suicide", "kill myself", "hopeless", "can't go on", "want to die"
}

