# rag.py
# Handles Retrieval-Augmented Generation using Pinecone.
# Re-added debug logging for Top 5 candidates + entry print.
import os
import re
from openai import OpenAI
from pinecone import Pinecone
import logging

# Configure logging (ensure level is INFO or DEBUG)
# If logs still missing, Render might override level; check Render settings.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize OpenAI and Pinecone Clients ---
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")
    if not client.api_key: raise ValueError("OPENAI_API_KEY missing.")
    if not pinecone_api_key: raise ValueError("PINECONE_API_KEY missing.")
    if not pinecone_index_name: raise ValueError("PINECONE_INDEX_NAME missing.")

    logging.info("Initializing Pinecone client...")
    pc = Pinecone(api_key=pinecone_api_key)
    logging.info(f"Checking if index '{pinecone_index_name}' exists...")
    if pinecone_index_name not in pc.list_indexes().names():
         raise ValueError(f"Pinecone index '{pinecone_index_name}' not found.")

    index = pc.Index(pinecone_index_name)
    logging.info(f"Successfully connected to Pinecone index '{pinecone_index_name}'.")
    # try: logging.info(f"Index stats: {index.describe_index_stats()}") # Keep commented unless debugging connection
    # except Exception as stats_e: logging.warning(f"Could not retrieve initial index stats: {stats_e}")

except Exception as e:
    logging.error(f"FATAL ERROR during API client initialization: {e}")
    client = None; index = None
    raise RuntimeError(f"Failed to initialize API clients: {e}") from e

# --- Core RAG Functions ---
def get_embedding(text, model="text-embedding-3-small"):
    # ... (code unchanged) ...
    if not client: logging.error("OpenAI client not initialized."); return None
    try:
        text = text.replace("\n", " ")
        response = client.embeddings.create(input=[text], model=model)
        if response.data and response.data[0].embedding: return response.data[0].embedding
        else: logging.error(f"Invalid OpenAI embeddings response: {response}"); return None
    except Exception as e: logging.error(f"Error getting embedding: {e}"); return None

def pinecone_search(query_embedding, faith_filter, top_k=5):
    # ... (code unchanged) ...
    if not index: logging.error("Pinecone index not initialized."); return []
    if not query_embedding: logging.error("Invalid query embedding."); return []
    if not faith_filter: logging.error("Faith filter cannot be empty."); return []
    source_filter_value = faith_filter
    try:
        logging.info(f"Querying Pinecone: filter={{'source': '{source_filter_value}'}}, top_k={top_k}")
        query_results = index.query(vector=query_embedding, filter={"source": source_filter_value}, top_k=top_k, include_metadata=True)
        logging.info(f"Pinecone returned {len(query_results.get('matches', []))} matches.")
        results_list = []
        if query_results and query_results.get('matches'):
            for match in query_results['matches']:
                metadata = match.get('metadata', {}); text = metadata.get('text'); ref = metadata.get('ref'); score = match.get('score', 0.0)
                if text and ref: results_list.append({"text": text, "ref": ref, "score": score})
                else: logging.warning(f"Match (ID: {match.get('id')}) missing metadata.")
            return results_list
        else: logging.info("No matches found in Pinecone."); return []
    except Exception as e: logging.error(f"Error querying Pinecone: {e}"); return []

# --- Helper Function ---
def clean_verse(text):
    # ... (code unchanged) ...
    if not text: return ""
    text = re.sub(r'\[\d+\]', '', text); text = text.replace("...", "").replace("..", ".")
    text = re.sub(r'\s+', ' ', text).strip(); return text

# --- Main Orchestration Function ---
def find_relevant_scripture(transformed_query: str, faith_context: str) -> tuple[str | None, str | None]:
    """Finds the single most relevant scripture using Pinecone."""
    # ★★★ Added Entry Print ★★★
    print("--- ENTERING find_relevant_scripture ---")
    logging.info(f"Starting Pinecone search: faith='{faith_context}', query='{transformed_query}'")
    if not transformed_query: logging.warning("Query empty."); return None, None
    if not faith_context: logging.warning("Faith context empty."); return None, None
    if not index or not client: logging.error("Clients not init."); return None, None

    query_embedding = get_embedding(transformed_query)
    if not query_embedding: logging.error("Failed query embedding."); return None, None

    search_results = pinecone_search(query_embedding, faith_context, top_k=5)

    # ★★★ Re-added DEBUGGING BLOCK ★★★
    if search_results:
        logging.info("--- Top 5 Pinecone Candidates ---")
        for i, result in enumerate(search_results):
            logging.info(f"{i+1}. Ref: {result.get('ref', 'N/A')}, Score: {result.get('score', 0.0):.4f}, Text: '{result.get('text', '')[:80]}...'")
        logging.info("--- End Candidates ---")
    else:
        logging.info("Pinecone returned no candidates.")
    # ★★★ END DEBUGGING BLOCK ★★★

    if search_results:
        for best_match in search_results: # Find first good one
            best_text = best_match.get('text'); best_ref = best_match.get('ref')
            if best_text and best_ref:
                cleaned_text = clean_verse(best_text)
                if len(cleaned_text.split()) >= 5: # Quality check
                    logging.info(f"Search complete. Selected valid match: {best_ref} (Score: {best_match.get('score', 0.0):.4f})")
                    return cleaned_text, best_ref
                else: logging.info(f"Skipping short match: {best_ref}")
            else: logging.warning("Match missing text/ref.")
        logging.info("No suitable long-enough match found in top results.")
        return None, None
    else:
        logging.info("No scripture found from Pinecone search.")
        return None, None

