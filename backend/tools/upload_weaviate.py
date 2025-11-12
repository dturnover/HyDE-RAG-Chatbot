import weaviate
import os
import json
import glob
from dotenv import load_dotenv

# --- SETTINGS ---
# ▼▼▼ Point this to your folder of JSONL files ▼▼▼
JSONL_FOLDER_PATH = "indexes" 
COLLECTION_NAME = "Scriptures"
# --- ---

# Load environment variables
load_dotenv()

# Get your cluster info
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not WEAVIATE_URL or not WEAVIATE_API_KEY:
    raise ValueError("WEAVIATE_URL or WEAVIATE_API_KEY not found in environment variables.")

print(f"Connecting to Weaviate at {WEAVIATE_URL}...")

try:
    # Connect to your Weaviate Cloud cluster
    client = weaviate.connect_to_wcs(
        cluster_url=WEAVIATE_URL,
        auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
        headers={
            "X-OpenAI-Api-Key": OPENAI_API_KEY
        }
    )

    # Get the collection we created in Step 2
    scriptures = client.collections.get(COLLECTION_NAME)
    print(f"Connected to collection '{COLLECTION_NAME}'.")

    # Use Weaviate's "batch" importer for high performance
    print("Starting batch import...")
    
    # Find all .jsonl files in the specified folder
    jsonl_files = glob.glob(os.path.join(JSONL_FOLDER_PATH, "*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in folder: {JSONL_FOLDER_PATH}")
        exit()

    print(f"Found {len(jsonl_files)} files to import: {jsonl_files}")

    total_count = 0
    with scriptures.batch.dynamic() as batch:
        # Loop through each file
        for file_path in jsonl_files:
            print(f"--- Processing file: {file_path} ---")
            count_in_file = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # 1. Separate the properties (metadata)
                        # We assume the 'source' is in the file.
                        # If not, we could derive it from the filename.
                        source = data.get("source")
                        if not source:
                            # Fallback: get source from filename
                            # e.g., "quran_clean_embed.jsonl" -> "quran"
                            source = os.path.basename(file_path).split('_')[0]

                        properties = {
                            "text": data.get("text"),
                            "ref": data.get("ref"),
                            "source": source
                        }
                        
                        # 2. Get the pre-calculated vector
                        vector = data.get("vector")
                        
                        if not vector or not properties["text"]:
                            print(f"Skipping bad line: {line[:50]}...")
                            continue

                        # 3. Add to the batch
                        batch.add_object(
                            properties=properties,
                            vector=vector  # This uploads your pre-calculated embedding
                        )
                        
                        count_in_file += 1
                        total_count += 1
                        
                        if count_in_file % 200 == 0:
                            print(f"Imported {count_in_file} objects from this file...")

                    except json.JSONDecodeError:
                        print(f"Skipping bad JSON line: {line[:50]}...")
            
            print(f"Finished processing {file_path}. Imported {count_in_file} objects.")

    print(f"\nSuccessfully imported a total of {total_count} objects from all files!")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if 'client' in locals() and client.is_connected():
        client.close()
        print("Connection closed.")