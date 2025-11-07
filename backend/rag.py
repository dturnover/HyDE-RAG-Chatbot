# rag.py
"""
This file is responsible for all "Retrieval-Augmented Generation" (RAG) tasks.
It's the bridge between our application and our two external services:
1.  OpenAI: Used to create vector embeddings (turning text into numbers).
2.  Pinecone: Our vector database, which stores and searches all the scriptures.
"""

import os
import re
from openai import OpenAI
from pinecone import Pinecone
import logging
import numpy as np  # Used for vector math (cosine similarity)
import random       # Used to pick a random result from top candidates

# Set up a basic logger for this file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize API Clients ---
# This code runs *once* when the server starts. It's critical that
# we connect to our services here, so the app will fail fast if an
# API key is missing, rather than failing on the first user's request.
try:
    # 1. Connect to OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # 2. Get Pinecone credentials from environment variables
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")
    
    # 3. Check that all keys were successfully loaded
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY missing from environment variables.")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY missing from environment variables.")
    if not pinecone_index_name:
        raise ValueError("PINECONE_INDEX_NAME missing from environment variables.")

    # 4. Connect to Pinecone
    logging.info("Initializing Pinecone client...")
    pc = Pinecone(api_key=pinecone_api_key)
    
    # 5. Check if the specific index (our database) exists
    logging.info(f"Checking if index '{pinecone_index_name}' exists...")
    if pinecone_index_name not in pc.list_indexes().names():
        raise ValueError(f"Pinecone index '{pinecone_index_name}' was not found.")

    # 6. Get the specific index object we'll be searching
    index = pc.Index(pinecone_index_name)
    logging.info(f"Successfully connected to Pinecone index '{pinecone_index_name}'.")
    
    try:
        # Log stats to confirm the connection is good
        logging.info(f"Index stats: {index.describe_index_stats()}")
    except Exception as stats_e:
        logging.warning(f"Could not retrieve initial index stats: {stats_e}")

except Exception as e:
    # If any connection fails, log the fatal error and stop the server.
    logging.error(f"FATAL ERROR during API client initialization: {e}")
    client = None
    index = None
    raise RuntimeError(f"Failed to initialize API clients: {e}") from e

# --- Core RAG Functions ---

def get_embedding(text: str, model="text-embedding-3-small") -> list[float] | None:
    """
    Converts a string of text into a vector embedding (a list of numbers)
    using OpenAI's embedding model.
    """
    if not client:
        logging.error("OpenAI client not initialized.")
        return None
    
    try:
        # OpenAI models prefer text with no newlines
        text = text.replace("\n", " ")
        
        # Call the OpenAI API
        response = client.embeddings.create(input=[text], model=model)
        
        # Extract the embedding from the response
        if response.data and response.data[0].embedding:
            return response.data[0].embedding
        else:
            logging.error(f"Invalid OpenAI embeddings response: {response}")
            return None
            
    except Exception as e:
        # This can happen if OpenAI's moderation filters block the input text
        logging.error(f"Error getting embedding: {e}", exc_info=True)
        return None

def pinecone_search(query_embedding: list[float], faith_filter: str, top_k=5) -> list:
    """
    Searches the Pinecone index for the 'top_k' most similar vectors
    to the query_embedding, filtering by the 'faith_filter' (e.g., "quran").
    """
    if not index:
        logging.error("Pinecone index not initialized.")
        return []
    if not query_embedding:
        logging.error("Invalid query embedding provided.")
        return []
    if not faith_filter:
        # This check is important; an empty filter would search the *entire* database
        logging.error("Faith filter cannot be empty.")
        return []
        
    try:
        # Query the Pinecone index
        query_results = index.query(
            vector=query_embedding,
            filter={"source": faith_filter},  # e.g., {"source": "bible_nrsv"}
            top_k=top_k,
            include_metadata=True  # This is critical to get the text and ref
        )
        
        # Format the raw Pinecone results into a simple list of dictionaries
        results_list = []
        if query_results and query_results.get('matches'):
            for match in query_results['matches']:
                # 'metadata' holds our scripture text and reference
                metadata = match.get('metadata', {})
                text = metadata.get('text')
                ref = metadata.get('ref')
                score = match.get('score', 0.0) # 'score' is the similarity
                
                # Only add the result if it has the data we need
                if text and ref:
                    results_list.append({"text": text, "ref": ref, "score": score})
                else:
                    logging.warning(f"A match (ID: {match.get('id')}) was missing 'text' or 'ref' in its metadata.")
            
            return results_list
        else:
            logging.info(f"No matches found in Pinecone for filter '{faith_filter}'.")
            return []
            
    except Exception as e:
        logging.error(f"Error querying Pinecone: {e}", exc_info=True)
        return []

# --- Helper Functions ---

def clean_verse(text: str) -> str:
    """
    A simple helper to remove common clutter (like footnote numbers)
    from the retrieved scripture text.
    """
    if not text:
        return ""
    
    # Removes [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    # Removes ellipses
    text = text.replace("...", "").replace("..", ".")
    # Replaces multiple spaces, newlines, tabs with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """
    Calculates the similarity (from -1 to 1) between two embedding vectors.
    This is used by the crisis-checking logic in logic.py.
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
        
        # Avoid division by zero
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        # The cosine similarity formula
        return dot_product / (norm_v1 * norm_v2)
        
    except Exception as e:
        logging.error(f"Error calculating cosine similarity: {e}", exc_info=True)
        return 0.0

# --- Main Orchestration Function ---

def find_relevant_scripture(transformed_query: str, faith_context: str) -> tuple[str | None, str | None]:
    """
    Finds the single most relevant scripture using Pinecone.
    
    This is the main function called by logic.py. It ties everything
    together: gets the embedding, searches Pinecone, and then
    randomly picks from the best results to avoid repetition.
    """
    # --- 1. Validation Checks ---
    if not transformed_query:
        logging.warning("find_relevant_scripture called with an empty query.")
        return None, None
    if not faith_context:
        logging.warning("find_relevant_scripture called with an empty faith.")
        return None, None
    if not index or not client:
        logging.error("RAG clients are not initialized, cannot find scripture.")
        return None, None

    # --- 2. Get Embedding ---
    try:
        query_embedding = get_embedding(transformed_query)
        if not query_embedding:
            logging.error(f"Failed to get embedding for query: '{transformed_query}', aborting search.")
            return None, None
    except Exception as e:
        logging.error(f"An unexpected error occurred during get_embedding: {e}", exc_info=True)
        return None, None

    # --- 3. Search Pinecone ---
    try:
        # Get the top 5 most similar scriptures
        search_results = pinecone_search(query_embedding, faith_context, top_k=5)
    except Exception as e:
        logging.error(f"An unexpected error occurred during pinecone_search: {e}", exc_info=True)
        return None, None

    # --- 4. Process Results ---
    if not search_results:
        logging.info(f"No scripture found from Pinecone search for query: '{transformed_query}'")
        return None, None
        
    # --- 5. Randomization Logic (to prevent repetition) ---
    # This logic prevents the bot from *always* returning the #1 result.
    # It finds all results "good enough" (95% of the top score) and
    # picks one at random.
    
    # 5a. Get the #1 score from the best match
    top_score = search_results[0].get('score', 0.0)
    
    # 5b. Set a threshold: any verse with a score at least 95%
    #     as good as the top score is a "candidate".
    score_threshold = top_score * 0.95 
    
    # 5c. Create a list of all "good enough" candidates
    candidates = []
    for match in search_results:
        if match.get('score', 0.0) >= score_threshold:
            candidates.append(match)
    
    # 5d. Find all *valid* candidates from that list (long enough, has text/ref)
    valid_candidates = []
    for match in candidates:
        text = match.get('text')
        ref = match.get('ref')
        
        if text and ref:
            cleaned_text = clean_verse(text)
            # Quality check: ensure the verse isn't just a tiny fragment
            if len(cleaned_text.split()) >= 5: 
                valid_candidates.append((cleaned_text, ref))
    
    # 5e. If we have valid, high-score candidates, pick one at random
    if valid_candidates:
        logging.info(f"RAG: Found {len(valid_candidates)} high-score candidates. Randomly selecting one.")
        return random.choice(valid_candidates)
        
    # --- 6. Fallback Logic ---
    # If the randomization logic failed (e.g., all top hits were too short),
    # just find the first good-enough match in the *entire* list.
    logging.warning("RAG: No valid *high-score* candidates found. Falling back to first available.")
    for best_match in search_results:
        best_text = best_match.get('text')
        best_ref = best_match.get('ref')
        
        if best_text and best_ref:
            cleaned_text = clean_verse(best_text)
            # Run the same quality check
            if len(cleaned_text.split()) >= 5: 
                return cleaned_text, best_ref
            else:
                logging.info(f"Skipping short match: {best_ref}")

    # If we get here, nothing was found at all
    logging.info(f"No suitable match found in top 5 results for: '{transformed_query}'")
    return None, None