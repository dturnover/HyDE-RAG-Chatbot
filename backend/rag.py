# rag.py
#
# This file handles the "Retrieval-Augmented Generation" (RAG) part.
# Its main jobs are:
# 1. Connecting to the OpenAI and Pinecone API services.
# 2. Creating a "vector embedding" (a number-based version) of a text query.
# 3. Searching the Pinecone vector database to find the most relevant scripture.
# 4. Providing utility functions like cosine similarity.

import os
import re
from openai import OpenAI
from pinecone import Pinecone
import logging  # We use logging for important messages and errors
import numpy as np  # ★★★ NEW: Added for cosine similarity math ★★★

# Set up the logger for this file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize OpenAI and Pinecone Clients ---

# We'll try to connect to the APIs right when the server starts.
# If this fails, the server will stop with an error, which is
# good because the app can't run without these connections.
try:
    # This is the OpenAI client, used for embeddings and chat
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # These are the details for our Pinecone vector database
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")
    
    # Check if the environment variables were actually found
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY missing from environment variables.")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY missing from environment variables.")
    if not pinecone_index_name:
        raise ValueError("PINECONE_INDEX_NAME missing from environment variables.")

    logging.info("Initializing Pinecone client...")
    pc = Pinecone(api_key=pinecone_api_key)
    
    logging.info(f"Checking if index '{pinecone_index_name}' exists...")
    if pinecone_index_name not in pc.list_indexes().names():
        raise ValueError(f"Pinecone index '{pinecone_index_name}' was not found.")

    # Connect to our specific index
    index = pc.Index(pinecone_index_name)
    logging.info(f"Successfully connected to Pinecone index '{pinecone_index_name}'.")
    
    try:
        # Just to be sure, print some stats about the index
        logging.info(f"Index stats: {index.describe_index_stats()}")
    except Exception as stats_e:
        logging.warning(f"Could not retrieve initial index stats: {stats_e}")

except Exception as e:
    logging.error(f"FATAL ERROR during API client initialization: {e}")
    # If we failed, set these to None so other files know
    client = None
    index = None
    # Stop the program from continuing
    raise RuntimeError(f"Failed to initialize API clients: {e}") from e

# --- Core RAG Functions ---

def get_embedding(text: str, model="text-embedding-3-small") -> list[float] | None:
    """
    Generates a vector "embedding" for a piece of text using OpenAI.
    An embedding is just a long list of numbers that represents
    the *meaning* of the text.
    """
    if not client:
        logging.error("OpenAI client not initialized.")
        return None
    try:
        # Replace newlines with spaces, as it's better for embedding models
        text = text.replace("\n", " ")
        response = client.embeddings.create(input=[text], model=model)
        
        if response.data and response.data[0].embedding:
            return response.data[0].embedding
        else:
            logging.error(f"Invalid OpenAI embeddings response: {response}")
            return None
    except Exception as e:
        logging.error(f"Error getting embedding: {e}", exc_info=True)
        return None

def pinecone_search(query_embedding: list[float], faith_filter: str, top_k=5) -> list:
    """
    Searches the Pinecone index for the most relevant scriptures.
    
    It uses the `query_embedding` (the numbers) to find the *closest*
    matches and filters the results by `faith_filter` (e.g., "quran", "bible_nrsv").
    """
    if not index:
        logging.error("Pinecone index not initialized.")
        return []
    if not query_embedding:
        logging.error("Invalid query embedding provided.")
        return []
    if not faith_filter:
        logging.error("Faith filter cannot be empty.")
        return []
        
    try:
        # This is the actual search query to Pinecone
        query_results = index.query(
            vector=query_embedding,
            filter={"source": faith_filter},  # Only search within this faith
            top_k=top_k,                     # Get the top 5 matches
            include_metadata=True            # We need this to get the text and reference
        )
        
        results_list = []
        if query_results and query_results.get('matches'):
            for match in query_results['matches']:
                # Pull the data out of the "metadata"
                metadata = match.get('metadata', {})
                text = metadata.get('text')
                ref = metadata.get('ref')
                score = match.get('score', 0.0)
                
                if text and ref:
                    results_list.append({"text": text, "ref": ref, "score": score})
                else:
                    logging.warning(f"A match (ID: {match.get('id')}) was missing 'text' or 'ref' in its metadata.")
            return results_list
        else:
            # This is a normal event, not an error.
            logging.info(f"No matches found in Pinecone for filter '{faith_filter}'.")
            return []
    except Exception as e:
        logging.error(f"Error querying Pinecone: {e}", exc_info=True)
        return []

# --- Helper Functions ---

def clean_verse(text: str) -> str:
    """
    A simple helper function to clean up scripture text.
    It removes things like [1] footnote numbers and extra spaces.
    """
    if not text:
        return ""
    text = re.sub(r'\[\d+\]', '', text)  # Remove [1], [2], etc.
    text = text.replace("...", "").replace("..", ".")
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with one
    return text

# ★★★ NEW: Cosine Similarity Function ★★★
def get_cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """
    Calculates the cosine similarity between two embedding vectors.
    Returns a score between -1 and 1 (or 0 and 1 for OpenAI embeddings).
    A score closer to 1 means the vectors are very similar in meaning.
    """
    try:
        # Convert lists to numpy arrays for efficient math
        v1_np = np.array(v1)
        v2_np = np.array(v2)
        
        # Calculate the dot product
        dot_product = np.dot(v1_np, v2_np)
        
        # Calculate the magnitudes (norms) of the vectors
        norm_v1 = np.linalg.norm(v1_np)
        norm_v2 = np.linalg.norm(v2_np)
        
        # Check for zero vectors to avoid division by zero
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        # Calculate the cosine similarity
        return dot_product / (norm_v1 * norm_v2)
    except Exception as e:
        logging.error(f"Error calculating cosine similarity: {e}", exc_info=True)
        return 0.0

# --- Main Orchestration Function ---

# ★★★ UPDATED with better error logging ★★★
def find_relevant_scripture(transformed_query: str, faith_context: str) -> tuple[str | None, str | None]:
    """
    This is the main function that ties everything in this file together.
    
    It takes the "antidote" query and the faith, gets the embedding,
    searches Pinecone, and returns the single best result.
    """
    if not transformed_query:
        logging.warning("find_relevant_scripture called with an empty query.")
        return None, None
    if not faith_context:
        logging.warning("find_relevant_scripture called with an empty faith.")
        return None, None
    if not index or not client:
        # This is a critical, app-breaking error
        logging.error("RAG clients are not initialized, cannot find scripture.")
        return None, None

    # 1. Turn the text query into numbers (embedding)
    try:
        query_embedding = get_embedding(transformed_query)
        if not query_embedding:
            # get_embedding already logs its own errors, but we'll log the failure.
            logging.error(f"Failed to get embedding for query: '{transformed_query}', aborting search.")
            return None, None
    except Exception as e:
        logging.error(f"An unexpected error occurred during get_embedding: {e}", exc_info=True)
        return None, None

    # 2. Search Pinecone
    try:
        search_results = pinecone_search(query_embedding, faith_context, top_k=5)
    except Exception as e:
        logging.error(f"An unexpected error occurred during pinecone_search: {e}", exc_info=True)
        return None, None

    # 3. Process results
    if not search_results:
        # This is not an error, it's just a "no match found."
        logging.info(f"No scripture found from Pinecone search for query: '{transformed_query}'")
        return None, None
        
    # Find the best, highest-quality match
    for best_match in search_results:
        best_text = best_match.get('text')
        best_ref = best_match.get('ref')
        
        if best_text and best_ref:
            cleaned_text = clean_verse(best_text)
            
            # Quality check for verse length
            if len(cleaned_text.split()) >= 5:
                return cleaned_text, best_ref
            else:
                # This is a useful debug log
                logging.info(f"Skipping short match (score: {best_match.get('score', 0):.2f}): {best_ref}")
        else:
            logging.warning("A search match was returned but was missing text or ref metadata.")
    
    # If we went through all 5 and none were good enough
    logging.info(f"No suitable (long-enough) match found in top 5 results for: '{transformed_query}'")
    return None, None