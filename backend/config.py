# config.py
"""
This file acts as the central settings and configuration hub for the entire 
chatbot application. All API keys, model names, and keyword lists are 
stored here for easy access and modification.
"""

import os
from dotenv import load_dotenv

# Load environment variables from a .env file (if one exists)
load_dotenv()

# --- API and Model Configuration ---
# Specifies which AI models we're using for generation and embedding.

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# A simple check to warn the developer if the API key is missing during startup
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY environment variable not found.")

# --- Faith Mapping ---
# This dictionary maps user-mentioned keywords to the correct "source"
# filter in our Pinecone vector database. This is how we know which
# religious text to search.
FAITH_KEYWORDS = {
    # Christian keywords (mapping to two different Bible versions)
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
    
    # Jewish keywords
    "jewish": "tanakh",
    "judaism": "tanakh",
    "hebrew": "tanakh",
    "tanakh": "tanakh",
    "torah": "tanakh",
    
    # Muslim keywords
    "muslim": "quran",
    "islam": "quran",
    "quran": "quran",
    "koran": "quran",
    
    # Hindu keywords
    "hindu": "gita",
    "gita": "gita",
    "bhagavad gita": "gita",
    "krishna": "gita",
    
    # Buddhist keywords
    "buddhist": "dhammapada",
    "buddhism": "dhammapada",
    "dhammapada": "dhammapada",
}

# --- RAG Trigger Keywords ---
# These sets determine *if* we should try to search Pinecone.

# DISTRESS_KEYWORDS: Emotional words.
# If a user's message contains one of these, it triggers our "HyDE"
# search (logic.py) to find an emotionally relevant passage.
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

# ASK_WORDS: Topic-based words.
# If a user's message contains one of these, it triggers a "normal"
# RAG search based on the user's message content.
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

# --- Layered Escalation Keywords ---
# These lists are for our 2-layer crisis detection system.

# LAYER 1: IMMEDIATE (Non-AI Check)
# This is our first line of defense. If these words are found (even with 
# typos), we *immediately* flag the session as 'crisis'.
#
# NOTE: "hopeless" was removed from this list. It is still in 
# DISTRESS_KEYWORDS, which allows the RAG system to find helpful 
# passages. The "I feel completely hopeless" phrase is still caught 
# by the Layer 2 semantic check below.
CRISIS_KEYWORDS_IMMEDIATE = {
    # Single words
    "suicide",
    
    # Key Phrases
    "kill myself",
    "can't go on",
    "want to die",
    "wanna die",
    "don't want to live anymore",
    "dont want to live anymore",
V    "don't want to be here anymore",
    "dont want to be here anymore",
    "want to end it all",
    "going to end it all"
}

# LAYER 2: SEMANTIC (AI Check)
# These are more subtle phrases. On startup, we create embeddings for
# these. We then check if the user's message is semantically similar
# to any of them. This catches phrases that Layer 1's keywords might miss.
CRISIS_PHRASES_SEMANTIC = [
    "I am suicidal",
    "My life isn't worth living",
    "I feel completely hopeless",
    "I am a burden to everyone",
    "The world is better off without me"
]