# config.py
#
# This file is the main configuration and settings hub for the application.
# This final version moves all critical crisis phrases into the
# "immediate" list to make the Layer 1 safety check robust.

import os
from dotenv import load_dotenv

load_dotenv()

# --- API and Model Configuration ---

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY environment variable not found.")

# --- Faith Mapping ---
FAITH_KEYWORDS = {
    "catholic": "bible_nrsv",
    "christian": "bible_asv",
    "protestant": "bible_asv",
    "baptist": "bible_asv",
    "methodist": "bible_asv",
    "lutheran": "bible_asv",
    "anglican": "bible_nrsv",
    "evangelical": "bible_asv",
    "orthodox": "bible_nrsv",
    "jesus": "bible_nrsv",
    "bible": "bible_nrsv",
    "asv": "bible_asv",
    "nrsv": "bible_nrsv",
    "king james": "bible_asv",
    "jewish": "tanakh",
    "judaism": "tanakh",
    "hebrew": "tanakh",
    "tanakh": "tanakh",
    "torah": "tanakh",
    "muslim": "quran",
    "islam": "quran",
    "quran": "quran",
    "koran": "quran",
    "hindu": "gita",
    "gita": "gita",
    "bhagavad gita": "gita",
    "krishna": "gita",
    "buddhist": "dhammapada",
    "buddhism": "dhammapada",
    "dhammapada": "dhammapada",
}

# --- RAG Trigger Keywords ---
DISTRESS_KEYWORDS = {
    "afraid", "alone", "anxiety", "anxious", "apprehensive", "dread",
    "fear", "frightened", "freaking out", "nervous", "on edge",
    "panicked", "panicking", "scared", "stress", "stressed", "tense",
    "terrified", "uneasy", "worried",
    "blue", "broken", "heartbroken", "depressed", "depression", "despair",
    "discouraged", "down", "grief", "grieving", "hopeless", "hurting",
    "low", "lonely", "lost", "miserable", "mourning", "sad", "sorrow",
    "unhappy",
    "angry", "annoyed", "betrayed", "bitter", "frustrated", "furious",
    "irritated", "jealous", "mad", "pissed", "rage", "resentful", "upset",
    "battling", "broken", "burnt out", "conflicted", "confused", "crushed",
    "defeated", "difficult", "drained", "empty", "exhausted", "failure",
    "failing", "guilt", "hard time", "insecure", "overwhelmed", "pain",
    "shame", "sin", "struggling", "stuck", "suffering", "tempted",
    "terrible", "tired", "trouble", "weak", "worn out"
}
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

# --- ★★★ UPDATED: Layered Escalation Keywords ★★★ ---

# LAYER 1: IMMEDIATE (Non-AI Check)
# This list is now much stronger and includes all critical phrases.
# Our typo-checker's regex will find these phrases anywhere in the user's message.
CRISIS_KEYWORDS_IMMEDIATE = {
    # Single words
    "suicide",
    "hopeless",
    
    # Key Phrases
    "kill myself",
    "can't go on",
    "want to die",
    "wanna die", # Added from our failed test
    "don't want to live anymore",
    "dont want to live anymore", # Added for typos
    "don't want to be here anymore",
    "dont want to be here anymore", # Added for typos
    "want to end it all",
    "going to end it all"
}

# LAYER 2: SEMANTIC (AI Check)
# This list is now for *subtler* phrases that our keyword
# list might miss. It's okay if this layer is blocked sometimes,
# because Layer 1 is now our main safety net.
CRISIS_PHRASES_SEMANTIC = [
    "I am suicidal",
    "My life isn't worth living",
    "I feel completely hopeless",
    "I am a burden to everyone",
    "The world is better off without me"
]