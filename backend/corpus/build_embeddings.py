
# build_embeddings.py
import os, json, tiktoken
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chunk_text(text, max_tokens=500, overlap=50):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = enc.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks

def embed_corpus(file_path, source, out_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    chunks = chunk_text(raw)

    out = []
    for i, chunk in enumerate(chunks):
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        out.append({
            "id": f"{source}-{i}",
            "text": chunk,
            "embedding": emb.data[0].embedding,
            "source": source
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(f"✅ Saved {len(out)} chunks to {out_path}")

# ---- run for your files ----
embed_corpus("kjv.txt", "bible", "bible.jsonl")
embed_corpus("quran-simple-plain.txt", "quran", "quran.jsonl")
embed_corpus("William Davidson Edition - English.txt", "talmud", "talmud.jsonl")
