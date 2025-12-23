# rag.py
#   This file is the engine/hands of the application. While logic.py is the
#   brain rag.py does the physical work: talking to OpenAI and Pinecone

import os
import re
from openai import OpenAI
from pinecone import Pinecone
import logging
import numpy as np
import random

# set up a basic logger for this file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- initialize API clients ---
# this code runs once when the server starts. It's critical that we connect
# to our services here, so the app will fail fast if an API key is missing,
# rather than failing on the first user's request

try:
    # 1. connect to OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # 2. get Pinecone credentials from environment variables
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")

    # 3. check that all keys were successfully loaded
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY missing from environment variables.")
    elif not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY missing from environment variables.")
    elif not pinecone_index_name:
        raise ValueError("PINECONE_INDEX_NAME missing from environment variables.")

    # 4. connect to Pinecone
    logging.info("Initializing Pinecone client...")
    pc = Pinecone(api_key=pinecone_api_key)

    # 5. check if the specific index (our database) exists
    logging.info(f"Checking if index '{pinecone_index_name}' exists...")
    if pinecone_index_name not in pc.list_indexes().names():
        raise ValueError(f"Pinecone index '{pinecone_index_name}' was not found.")

    # 6. get the specific index object we'll be searching
    index = pc.Index(pinecone_index_name)
    logging.info(f"Successfully connected to Pinecone index '{pinecone_index_name}'.")

    try:
        # Log stats to confirm the connection is good
        logging.info(f"Index stats: {index.describe_index_stats()}")
    except Exception as stats_e:
        logging.warning(f"Could not retrieve initial index stats: {stats_e}")

except Exception as e:
    # if any connection fails log the fatal error and stop the server
    logging.error(f"FATAL ERROR during API client initialization: {e}")
    client = None
    index = None
    raise RuntimeError(f"Failed to initialize API clients: {e}") from e



# --- core RAG functions ---

def get_embedding(text, model="text-embedding-3-small"):
    # converts a string into a vector embedding (list of numbers) using OpenAI's embedding model

    if not client:
        logging.error("OpenAI client not initialized.")
        return None 

    try:
        # OpenAI models prefer text with no newlines
        text = text.replace("\n", " ")

        # call the OpenAI API
        response = client.embeddings.create(input=[text], model=model)

        # extract the embeddings from the response
        if response.data and response.data[0].embedding:
            return response.data[0].embedding
        else:
            logging.error(f"Invalid OpenAI embeddings response: {response}")
            return None 

    except Exception as e:
        # this can happen if OpenAI's moderation filters block the input text
        logging.error(f"Error getting embedding: {e}", exc_info=True)
        return None 


def pinecone_search(query_embedding, faith_filter, top_k=5):
    # searches the Pinecone index for the top k most similar vectors to the query_embedding,
    # filtering by the 'faith_filter' (e.g. "quran")

    if not index:
        logging.error("Pinecone index not initialized.")
        return []
    if not query_embedding:
        logging.error("Invalid query embedding provided.")
        return []

    try:
        # Query the Pinecone index
        query_results = index.query(
            vector=query_embedding,
            filter={"source": faith_filter},
            top_k=top_k,
            include_metadata=True   
        )

        # format the raw pinecone results into a simple list of dictionaries
        results_list = []
        if query_results and query_results.get('matches'):
            for match in query_results['matches']:
                # 'metadata' holds our scripture text and reference
                metadata = match.get('metadata', {})
                text = metadata.get('text')
                ref = metadata.get('ref')
                score = match.get('score', 0.0) # 'score' is the similarity

                # noly add the result if it has the data we need
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



# --- helper functoins ---

def clean_verse(text):
    # simple helper to remove common clutterlike footnote numbers from scripture text

    if not text:
        return ""

    # removed [1], [2], etc
    text = re.sub(r'\[\d+\]', '', text)
    # removes ellipses
    text = text.replace("...", "").replace("..", ".")
    # replaces multiple spaces, newlines, tabs with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text



# --- main orchestration function ---

def find_relevant_scripture(transformed_query, faith_context):
    # finds the single most relevant scripture using Pincone
    #   this is the main function called by logic.py. It ties everything together
    #   gets the embeddings, searches Pinecone, and then randomly picks the best results to avoid repetitions

    # --- 1. validation checks ---
    if not transformed_query:
        logging.warning("find_relevant_scripture called with an empty query.")
        return None, None 
    if not faith_context:
        logging.warning("find_relevant_scripture called with an empty faith.")
        return None, None 
    if not index or not client:
        logging.error("RAG clients are not initialized, cannot find scripture.")
        return None, None 

    # --- 2. get embedding ---
    try:
        query_embedding = get_embedding(transformed_query)
        if not query_embedding:
            logging.error(f"Failed to get embedding for query: '{transformed_query}', aborting search.")
            return None, None 
    except Exception as e:
        logging.error(f"An unexpected error occurred during get_embedding: {e}", exc_info=True)
        return None, None 

    # --- 3. search pinecone ---
    try:
        # get the top 5 most similar scriptures
        search_results = pinecone_search(query_embedding, faith_context, top_k=5)
    except Exception as e:
        logging.error(f"An unexpected error occurred during pinecone_search: {e}", exc_info=True)
        return None, None 

    # --- 4. process results ---
    if not search_results:
        logging.info(f"No scripture found from Pinecone search for query: ' {transformed_query}'")
        return None, None 

    # -- 5. randomization logic (to prevent repetition) ---

    # 5a. get the #1 score from the best match
    top_score = search_results[0].get('score', 0.0)

    # 5b. set a threshold: any verse with a score at least 95%
    score_threshold = top_score * 0.95

    # 5c. create a list of all good enough candidates
    candidates = []
    for match in search_results:
        if match.get('score', 0.0) >= score_threshold:
            candidates.append(match)

    # 5d. find all valid candidates from that list
    valid_candidates = []
    for match in candidates:
        text = match.get('text')
        ref = match.get('ref')

        if text and ref:
            cleaned_text = clean_verse(text)
            # quality check: ensure the verse isn't jsut a tiny fragment
            if len(cleaned_text.split()) >= 5:
                valid_candidates.append((cleaned_text, ref))

    # 5e. if we have valid, high-score candidates, pick one at random
    if valid_candidates:
        logging.info(f"RAG: Found {len(valid_candidates)} high-score candidates. Randomly selecting one.")
        return random.choice(valid_candidates)

    # --- 6. fallback logic ---
    #   if the randomization logic failed just find the first good-enough match
    logging.warning("RAG: Novalid *high-score* candidates found. Falling back to first available.")
    for best_match in search_results:
        best_text = best_match.get('text')
        best_ref = best_match.get('ref')

        if best_text and best_ref:
            cleaned_text = clean_verse(best_text)
            # run the same quality check
            if len(cleaned_text.split()) >= 5:
                return cleaned_text, best_ref
            else:
                logging.info(f"Skipping short match: {best_ref}")

    # if we get here nothing was found at all
    logging.info(f"No suitable match was found in top 5 results for: '{transformed_query}'")
    return None, None
