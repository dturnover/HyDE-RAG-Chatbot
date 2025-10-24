import os
import sys
import json
from openai import OpenAI
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
BATCH = int(os.getenv("EMBED_BATCH", "64"))
# A safe character limit. OpenAI's token limit is 8192 tokens (~32k chars).
# We'll set a very safe limit here to prevent any oversized requests.
CHAR_LIMIT = 16000

def write_batch(cli: OpenAI, texts: list[str], metas: list[dict], out):
    """Gets embeddings for a batch of texts and writes them to the output file."""
    try:
        resp = cli.embeddings.create(model=MODEL, input=texts)
        for meta, e in zip(metas, resp.data):
            meta["embedding"] = e.embedding
            out.write(json.dumps(meta, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"\nAn error occurred during embedding batch: {e}")
        # Skipping this failed batch.

def main(inp: str, outp: str):
    """
    Reads a raw .jsonl file, generates embeddings for the 'text' field in batches,
    and writes the result to a new _embed.jsonl file.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)
        
    cli = OpenAI(api_key=api_key)
    
    try:
        with open(inp, "r", encoding="utf-8") as f_in, \
             open(outp, "w", encoding="utf-8") as f_out:
            
            buf, metas = [], []
            lines = f_in.readlines()
            
            for i, line in enumerate(tqdm(lines, desc=f"Processing {os.path.basename(inp)}")):
                line = line.strip()
                if not line: continue
                
                try:
                    m = json.loads(line)
                    text_to_embed = m.get("text")

                    if not text_to_embed:
                        continue

                    # ★★★ THE FIX IS HERE: A "SAFETY VALVE" ★★★
                    # Check the length of the text before adding it to the batch.
                    if len(text_to_embed) > CHAR_LIMIT:
                        print(f"\nWARNING: Skipping oversized text chunk on line {i+1} of {os.path.basename(inp)} (Ref: {m.get('ref', 'N/A')}).")
                        continue # Skip this oversized line

                    metas.append(m)
                    buf.append(text_to_embed)
                    
                    if len(buf) >= BATCH:
                        write_batch(cli, buf, metas, f_out)
                        buf, metas = [], []

                except json.JSONDecodeError:
                    print(f"\nWARNING: Skipping malformed JSON on line {i+1} of {os.path.basename(inp)}.")
                    continue

            # Process the final batch if any text is left in the buffer
            if buf:
                write_batch(cli, buf, metas, f_out)

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python tools/embed_jsonl.py <in.jsonl> <out.jsonl>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
