import os, sys, json
from openai import OpenAI

MODEL = os.getenv("EMBED_MODEL","text-embedding-3-small")

def main(inp, outp):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    out = open(outp, "w", encoding="utf-8")
    with open(inp, "r", encoding="utf-8") as f:
        buf, objs = [], []
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            objs.append(obj); buf.append(obj["text"])
            if len(buf) == 1000:  # batch safely
                write_batch(client, buf, objs, out); buf, objs = [], []
        if buf: write_batch(client, buf, objs, out)
    out.close()

def write_batch(client, texts, objs, out):
    emb = client.embeddings.create(model=MODEL, input=texts)
    for obj, e in zip(objs, emb.data):
        obj["embedding"] = e.embedding
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    if len(sys.argv)<3:
        print("usage: python embed_jsonl.py <in.jsonl> <out.jsonl>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
