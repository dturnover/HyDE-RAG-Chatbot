# rag.py
# Handles Retrieval-Augmented Generation using Pinecone.
# Confirmed to align with baton pass instructions.
import os
import re
from openai import OpenAI
from pinecone import Pinecone # Added import
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize OpenAI and Pinecone Clients ---
# Important: Ensure OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_INDEX_NAME
# are set as environment variables in your deployment environment (e.g., Render).
try:
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")

    if not openai_client.api_key:
         raise ValueError("OPENAI_API_KEY environment variable not set or invalid.")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set.")
    if not pinecone_index_name:
        raise ValueError("PINECONE_INDEX_NAME environment variable not set.")

    # Initialize Pinecone connection
    logging.info("Initializing Pinecone client...")
    pc = Pinecone(api_key=pinecone_api_key)

    # Validate index connection during initialization
    logging.info(f"Checking if index '{pinecone_index_name}' exists...")
    if pinecone_index_name not in pc.list_indexes().names():
         # Raise an error if it's expected to exist
         raise ValueError(f"Pinecone index '{pinecone_index_name}' not found in the account. Please ensure it's created and the name is correct.")

    index = pc.Index(pinecone_index_name)
    logging.info(f"Successfully connected to Pinecone index '{pinecone_index_name}'.")
    # Log initial stats for confirmation
    try:
        logging.info(f"Index stats: {index.describe_index_stats()}")
    except Exception as stats_e:
        logging.warning(f"Could not retrieve initial index stats: {stats_e}")


except Exception as e:
    logging.error(f"FATAL ERROR during API client initialization: {e}")
    openai_client = None
    index = None
    # For a server, it's often better to raise the error to prevent startup
    # rather than continuing in a non-functional state.
    raise RuntimeError(f"Failed to initialize API clients: {e}") from e


# --- Core RAG Functions ---

def get_embedding(text, model="text-embedding-3-small"):
    """Generates an embedding for the given text using OpenAI."""
    if not openai_client:
        logging.error("OpenAI client not initialized. Cannot get embedding.")
        return None
    try:
        text = text.replace("\n", " ")
        response = openai_client.embeddings.create(input=[text], model=model)
        if response.data and response.data[0].embedding:
             return response.data[0].embedding
        else:
             logging.error(f"Invalid response from OpenAI embeddings API: {response}")
             return None
    except Exception as e:
        logging.error(f"Error getting embedding for text '{text[:50]}...': {e}")
        return None

def pinecone_search(query_embedding, faith_filter, top_k=5): # Using top_k=5 as requested
    """
    Searches the Pinecone index for the most relevant scriptures.
    Returns a list of results.

    Args:
        query_embedding (list[float]): The vector embedding of the search query.
        faith_filter (str): The 'source' value to filter by (e.g., 'bible_nrsv', 'quran').
        top_k (int): The number of results to return.

    Returns:
        list[dict]: A list of dictionaries, each containing 'text', 'ref', and 'score' for a match,
                    or an empty list if error/no match.
    """
    if not index:
        logging.error("Pinecone index not initialized. Cannot perform search."); return []
    if not query_embedding:
        logging.error("Invalid query embedding provided."); return []
    if not faith_filter:
        logging.error("Faith filter cannot be empty."); return []

    # Map simple faith names to the actual source names used as IDs/metadata
    # This might need adjustment based on how FAITH_KEYWORDS maps in config.py
    # Assuming config.FAITH_KEYWORDS maps "christian" -> "bible_nrsv", "jewish" -> "tanakh" etc.
    # If the faith_filter *is* already the source name (e.g., "bible_nrsv"), this mapping isn't strictly needed here.
    # Let's assume faith_filter IS the source name for now based on previous context.
    source_filter_value = faith_filter
    # Example mapping if needed later:
    # faith_to_source_map = {"christian": "bible_nrsv", "jewish": "tanakh", ...}
    # source_filter_value = faith_to_source_map.get(faith_filter.lower(), faith_filter) # Fallback to original if not mapped


    try:
        logging.info(f"Querying Pinecone index '{pinecone_index_name}' with filter={{'source': '{source_filter_value}'}}, top_k={top_k}")
        query_results = index.query(
            vector=query_embedding,
            filter={
                "source": source_filter_value # Filter by the 'source' metadata field
            },
            top_k=top_k,
            include_metadata=True # Essential to get text and ref
        )
        logging.info(f"Pinecone query returned {len(query_results.get('matches', []))} matches.")

        results_list = []
        if query_results and query_results.get('matches'):
            for match in query_results['matches']:
                metadata = match.get('metadata', {})
                text = metadata.get('text')
                reference = metadata.get('ref')
                score = match.get('score', 0.0) # Cosine similarity score

                if text and reference:
                    # logging.debug(f"Match found: Ref='{reference}', Score={score:.4f}, Text='{text[:50]}...'") # Debug level
                    results_list.append({"text": text, "ref": reference, "score": score})
                else:
                    logging.warning(f"Match (ID: {match.get('id')}) missing text or reference in metadata.")
            # Sort results by score (Pinecone usually does, but good practice)
            results_list.sort(key=lambda x: x['score'], reverse=True)
            return results_list
        else:
            logging.info("No relevant matches found in Pinecone.")
            return []

    except Exception as e:
        logging.error(f"Error querying Pinecone: {e}")
        return []


# --- Helper Function ---
def clean_verse(text):
    """Removes extraneous characters and formatting from verse text."""
    if not text: return ""
    text = re.sub(r'\[\d+\]', '', text) # Footnotes
    text = text.replace("...", "").replace("..", ".") # Ellipses
    text = re.sub(r'\s+', ' ', text).strip() # Whitespace
    return text


# --- Main Orchestration Function ---
def find_relevant_scripture(transformed_query: str, faith_context: str) -> tuple[str | None, str | None]:
    """
    Finds the single most relevant scripture using embedding search via Pinecone.

    Args:
        transformed_query (str): The query generated by the LLM rewrite step.
        faith_context (str): The faith source to search within (e.g., 'bible_nrsv').

    Returns:
        tuple[str | None, str | None]: A tuple containing (verse_text, verse_ref) of the BEST match,
                                      or (None, None) if no relevant scripture is found or an error occurs.
    """
    logging.info(f"Starting Pinecone scripture search for faith '{faith_context}' with query: '{transformed_query}'")
    if not transformed_query: logging.warning("Query empty."); return None, None
    if not faith_context: logging.warning("Faith context empty."); return None, None
    if not index or not openai_client: logging.error("Clients not init."); return None, None

    # 1. Get embedding
    query_embedding = get_embedding(transformed_query)
    if not query_embedding: logging.error("Failed to get query embedding."); return None, None

    # 2. Search Pinecone - Get top results (pinecone_search returns sorted list)
    search_results = pinecone_search(query_embedding, faith_context, top_k=5) # Request 5

    # 3. Select the best valid result
    if search_results:
        # Iterate through results to find the first one that passes quality checks
        for best_match in search_results:
            best_text = best_match.get('text')
            best_ref = best_match.get('ref')

            if best_text and best_ref:
                cleaned_text = clean_verse(best_text)
                # ★★★ Add quality check: ensure text is reasonably long ★★★
                if len(cleaned_text.split()) >= 5: # Check for minimum word count
                    logging.info(f"Search complete. Best valid match: {best_ref} (Score: {best_match.get('score', 0.0):.4f})")
                    return cleaned_text, best_ref
                else:
                     logging.info(f"Skipping short match: {best_ref} (Text: '{cleaned_text[:50]}...')")
            else:
                 logging.warning("Match from Pinecone was missing text or ref.")

        # If loop finishes without finding a good match
        logging.info("No suitable long-enough scripture found among top Pinecone results.")
        return None, None
    else:
        logging.info("No scripture found from Pinecone search.")
        return None, None

# --- Old file-based functions (DELETED) ---
# load_embeddings, cosine_similarity, semantic_search,
# keyword_search, hybrid_search, tokenize

