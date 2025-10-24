# tools/upload_to_pinecone.py
# A one-time script to upload the final _embed.jsonl files to a Pinecone index.
#
# INSTRUCTIONS:
# 1. (RECOMMENDED) Set your environment variables before running:
#    - For Windows PowerShell:
#      $Env:PINECONE_API_KEY="your-api-key"
#      $Env:PINECONE_INDEX_NAME="your-index-name"
#    - For macOS/Linux:
#      export PINECONE_API_KEY="your-api-key"
#      export PINECONE_INDEX_NAME="your-index-name"
# 2. If you don't set environment variables, you will be prompted to enter them.
# 3. Run this script for each of your six _embed.jsonl files.

import json
import argparse
import os
from pinecone import Pinecone
from getpass import getpass

def upload_to_pinecone(file_path: str):
    """
    Reads a .jsonl file with embeddings and uploads them to a Pinecone index.
    """
    # --- 1. Get Pinecone Credentials ---
    print("--- Pinecone Configuration ---")
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")

    if not api_key:
        print("PINECONE_API_KEY environment variable not found.")
        api_key = getpass("Enter your Pinecone API Key: ")
    else:
        print("Found PINECONE_API_KEY environment variable.")

    if not index_name:
        print("PINECONE_INDEX_NAME environment variable not found.")
        index_name = input("Enter your Pinecone index name: ")
    else:
        print(f"Found PINECONE_INDEX_NAME: '{index_name}'")


    if not api_key or not index_name:
        print("API Key and index name are required. Exiting.")
        return

    # --- 2. Initialize Pinecone ---
    try:
        pc = Pinecone(api_key=api_key)
        
        if index_name not in pc.list_indexes().names():
            print(f"Error: Index '{index_name}' does not exist in your project.")
            print("Please create it in the Pinecone console with dimension 1536 and metric 'cosine'.")
            return

        index = pc.Index(index_name)
        print("Successfully connected to Pinecone index.")
        stats = index.describe_index_stats()
        print(f"Initial index state: {stats}")

    except Exception as e:
        print(f"Error connecting to Pinecone: {e}")
        return

    # --- 3. Read and Upload Data in Batches ---
    batch_size = 100
    batch = []
    total_vectors_uploaded = 0

    print(f"\n--- Starting Upload for {file_path} ---")
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                
                vector_to_upsert = {
                    "id": record["id"],
                    "values": record["embedding"],
                    "metadata": {
                        "text": record["text"],
                        "ref": record["ref"],
                        "source": record.get("source", "unknown") # Use .get for safety
                    }
                }
                batch.append(vector_to_upsert)

                if len(batch) >= batch_size:
                    print(f"Uploading batch of {len(batch)} vectors...")
                    index.upsert(vectors=batch)
                    total_vectors_uploaded += len(batch)
                    batch = []

            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON on line {i+1}")
            except KeyError as e:
                print(f"Warning: Skipping record on line {i+1} due to missing key: {e}")

    if batch:
        print(f"Uploading final batch of {len(batch)} vectors...")
        index.upsert(vectors=batch)
        total_vectors_uploaded += len(batch)

    print("\n--- Upload Complete ---")
    print(f"Total vectors uploaded from {file_path}: {total_vectors_uploaded}")
    stats = index.describe_index_stats()
    print(f"Final index state: {stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload embedded JSONL data to Pinecone.")
    parser.add_argument("file_path", help="Path to the _embed.jsonl file.")
    args = parser.parse_args()
    upload_to_pinecone(args.file_path)

