# config.py
import os
from pathlib import Path

# --- API and Model Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# --- RAG Index Configuration ---
RAG_DATA_PATH = Path("/opt/render/project/src/indexes")

# --- DEBUG STATEMENTS ---
print("\n--- [DEBUG] Loading config.py ---")
print(f"[DEBUG config.py] OpenAI Model: {OPENAI_MODEL}")
print(f"[DEBUG config.py] RAG_DATA_PATH set to: {RAG_DATA_PATH}")
print(f"[DEBUG config.py] Checking path existence...")
try:
    path_exists = RAG_DATA_PATH.exists()
    is_directory = RAG_DATA_PATH.is_dir()
    print(f"[DEBUG config.py] -> Does the path exist? {path_exists}")
    print(f"[DEBUG config.py] -> Is it a directory? {is_directory}")
    
    if path_exists and is_directory:
        print("[DEBUG config.py] SUCCESS: RAG data path is valid.")
        
        # ★★★ NEW: List the contents of the directory ★★★
        print("[DEBUG config.py] Listing contents of RAG data path:")
        found_files = False
        for item in RAG_DATA_PATH.iterdir():
            print(f"[DEBUG config.py]   - Found: {item.name}")
            found_files = True
        if not found_files:
            print("[DEBUG config.py]   - WARNING: Directory is empty.")
            
    else:
        print("[DEBUG config.py] ERROR: RAG data path is invalid or not found.")
except Exception as e:
    print(f"[DEBUG config.py] ERROR: An exception occurred while checking the path: {e}")
print("--- [DEBUG] Finished loading config.py ---\n")
# --- END DEBUG STATEMENTS ---

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