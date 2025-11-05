# config.py
#
# This file is the main configuration and settings hub for the application.
# It holds API keys, model names, and all the important keyword lists
# that the chatbot logic uses to make decisions.

import os
from dotenv import load_dotenv  # Used to load settings from a .env file

# This command looks for a file named ".env" in your project
# and loads any variables inside it into the environment.
# This is a good way to keep secret API keys out of your code.
load_dotenv()

# --- API and Model Configuration ---

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY environment variable not found.")

# --- Faith Mapping ---
# (This section is unchanged)
FAITH_KEYWORDS = {
    # Catholic uses the NRSV translation
    "catholic": "bible_nrsv",
    
    # Other Christian denominations use the ASV translation
    "christian": "bible_asv",
    "protestant": "bible_asv",
    "baptist": "bible_asv",
    "methodist": "bible_asv",
    "lutheran": "bible_asv",
    "anglican": "bible_nrsv",
    "evangelical": "bible_asv",
    "orthodox": "bible_nrsv",  # Eastern Orthodox
    "jesus": "bible_nrsv",
    "bible": "bible_nrsv",  # Default Bible to NRSV
    "asv": "bible_asv",  # Specific version
    "nrsv": "bible_nrsv",  # Specific version
    "king james": "bible_asv",

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
# (This section is unchanged)

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
# These are high-lethality keywords. We check these *first*
# with the typo-tolerance function. This check is fast, free,
# and CANNOT be blocked by OpenAI's moderation.
CRISIS_KEYWORDS_IMMEDIATE = {
    "suicide", "kill myself", "hopeless", "can't go on", "want to die"
}

# LAYER 2: SEMANTIC (AI Check)
# This is the list of full phrases for our *semantic* check.
# This will only run if the LAYER 1 check passes.
CRISIS_PHRASES_SEMANTIC = [
    "I am suicidal",
    "I want to kill myself",
    "I want to die",
    "I don't want to live anymore",
    "I don't want to be here anymore",
    "I feel completely hopeless",
    "I can't go on",
    "My life isn't worth living",
    "I'm going to end it all"
]