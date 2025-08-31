import os, sys, json
from openai import OpenAI

MODEL = os.getenv("EMBED_MODEL","text-embedding-3-small")
BATCH = int(os.getenv("EMBED_BATCH","64"))

def write_batch(cli, texts, metas, out):
    resp = cli.embeddings.create(model=MODEL, input=texts)
    for meta, e in zip(metas, resp.data):
        meta["embedding"] = e.embedding
        out.write(json.dumps(meta, ensure_ascii=False) + "\n")

def main(inp, outp):
    cli = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    out = open(outp, "w", encoding="utf-8")
    buf, metas = [], []
    with open(inp, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            m = json.loads(line)
            metas.append(m); buf.append(m["text"])
            if len(buf) >= BATCH:
                write_batch(cli, buf, metas, out); buf, metas = [], []
    if buf: write_batch(cli, buf, metas, out)
    out.close()

if __name__ == "__main__":
    if len(sys.argv)<3:
        print("usage: python tools/embed_jsonl.py <in.jsonl> <out.jsonl>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
