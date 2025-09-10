# backend/app.py (patched)
import os, re, json, uuid, math, asyncio
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# --- OpenAI client ---
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False
    OpenAI = None  # type: ignore

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

def oa_client() -> Optional[OpenAI]:
    if not (_HAS_OPENAI and OPENAI_API_KEY):
        return None
    return OpenAI(api_key=OPENAI_API_KEY)

def embed_query(text: str) -> Optional[List[float]]:
    cli = oa_client()
    if not cli: return None
    try:
        r = cli.embeddings.create(model=EMBED_MODEL, input=text)
        return r.data[0].embedding  # type: ignore
    except Exception:
        return None

def call_openai(messages: List[Dict[str,str]], stream: bool):
    cli = oa_client()
    if not cli:
        return "⚠️ OpenAI disabled on server."
    if stream:
        resp = cli.chat.completions.create(model=OPENAI_MODEL, messages=messages,
                                           temperature=0.4, stream=True)
        def _gen():
            for ev in resp:
                piece = ev.choices[0].delta.content or ""
                if piece: yield piece
        return _gen()
    else:
        resp = cli.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.4)
        return resp.choices[0].message.content or ""

# --- Session state ---
class SessionState:
    def __init__(self):
        self.history = [{
            "role":"system",
            "content":(
                "You are Fight Chaplain: calm, concise, encouraging.\n"
                "Follow 7-step process: greet, assess emotional tone, reflect scripture "
                "via retrieval (never hallucinate verses), track session, escalate if crisis, "
                "refer to BetterHelp if needed, and always offer a faith leader referral.\n"
                "Keep answers tight and grounded. Only quote verses passed in RETRIEVED PASSAGES."
            )
        }]
        self.rollup: Optional[str] = None
        self.faith: Optional[str] = None

SESSIONS: Dict[str,SessionState] = {}
def get_or_create_sid(request: Request, response: Response) -> Tuple[str,SessionState]:
    sid = request.headers.get("X-Session-Id") or request.cookies.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        response.set_cookie("sid", sid, httponly=True, samesite="none", secure=True, path="/")
    if sid not in SESSIONS:
        SESSIONS[sid] = SessionState()
    return sid, SESSIONS[sid]

# --- Index loading helpers ---
def _candidate_index_dirs() -> List[Path]:
    here = Path(__file__).parent.resolve()
    env = os.getenv("RAG_DIR", "").strip()
    cands = []
    if env: cands.append(Path(env))
    cands += [here/"indexes", here/"backend"/"indexes", Path("/opt/render/project/src/indexes")]
    seen, out = set(), []
    for p in cands:
        if str(p) not in seen: out.append(p); seen.add(str(p))
    return out

def load_jsonl(path: Path) -> List[Dict[str,Any]]:
    rows = []
    if not path.exists(): return rows
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try: o = json.loads(line.strip())
            except: continue
            if not o.get("text") or not o.get("embedding"): continue
            rows.append({
                "id": o.get("id") or "",
                "ref": o.get("ref") or "",
                "source": o.get("source") or "",
                "text": o["text"],
                "embedding": o["embedding"],
            })
    return rows

def find_and_load_corpora() -> Dict[str,List[Dict[str,Any]]]:
    corpora = {"bible": [], "quran": [], "talmud": []}
    for base in _candidate_index_dirs():
        for name in ("bible.jsonl","quran.jsonl","talmud.jsonl"):
            p = base/name
            if p.exists() and p.stat().st_size>0:
                key = name.split(".")[0]
                if not corpora[key]:
                    corpora[key] = load_jsonl(p)
    return corpora

CORPORA = find_and_load_corpora()

# --- Simple utils ---
def cos(a: List[float], b: List[float]) -> float:
    s=na=nb=0.0
    for x,y in zip(a,b): s+=x*y; na+=x*x; nb+=y*y
    return s/math.sqrt(na*nb) if na and nb else 0.0

def tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower())

def jaccard(a:set,b:set)->float:
    return len(a&b)/len(a|b) if a and b else 0.0

# --- Faith routing ---
FAITH_KEYWORDS = {
    "muslim":"quran","islam":"quran","quran":"quran",
    "jew":"talmud","talmud":"talmud","hebrew":"talmud",
    "christian":"bible","catholic":"bible","bible":"bible"
}
def try_set_faith(msg: str, s: SessionState):
    m = msg.lower()
    for k,v in FAITH_KEYWORDS.items():
        if k in m: s.faith=v; return

def detect_corpus(msg: str, s: SessionState)->str:
    return s.faith or (
        "quran" if "quran" in msg.lower() else
        "talmud" if "talmud" in msg.lower() else "bible"
    )

# --- Hybrid search ---
def hybrid_search(query: str, corpus_name: str, top_k=6):
    docs = CORPORA.get(corpus_name,[])
    if not docs: return []
    q_tokens=set(tokenize(query))
    lex_scores=[(jaccard(q_tokens,set(tokenize(d["text"]))),i) for i,d in enumerate(docs)]
    vec = embed_query(query)
    vec_scores=[(cos(vec,d["embedding"]),i) for i,d in enumerate(docs)] if vec else []
    def z(lst): 
        if not lst: return {}
        mu=sum(x for x,_ in lst)/len(lst)
        sd=math.sqrt(sum((x-mu)**2 for x,_ in lst)/len(lst)) or 1
        return {j:(x-mu)/sd for x,j in lst}
    L,V=z(lex_scores),z(vec_scores)
    blend=[(0.6*L.get(i,0)+0.4*V.get(i,0),i) for i in range(len(docs))]
    blend.sort(reverse=True)
    return [docs[i] for _,i in blend[:top_k]]

def clean_citation(hit: Dict[str,Any]) -> str:
    ref = hit.get("ref","").strip()
    return ref.replace("p","") if ref else ""

# --- App setup ---
app = FastAPI()
ALLOWED_ORIGINS={"https://dturnover.github.io","http://localhost:3000"}
app.add_middleware(CORSMiddleware, allow_origins=list(ALLOWED_ORIGINS),
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/health") def health(): return PlainTextResponse("OK")
@app.get("/diag_rag") def diag(): return {"ok":True,"corpora":{k:len(v) for k,v in CORPORA.items()}}

# --- Chat endpoints ---
def build_messages(s:SessionState,user_msg:str,hits:List[Dict[str,Any]]):
    sys=_sys_with_rollup(s)
    if hits:
        verse=hits[0]["text"].strip().replace("\n"," ")
        ref=clean_citation(hits[0])
        ctx=(f"RETRIEVED PASSAGE:\n“{verse}” — {ref}\n\n"
             "In your answer, weave the passage naturally into supportive guidance. "
             "Do not invent references; only use the above.")
        sys["content"]+="\n\n"+ctx
    return [sys]+s.history[1:]+[{"role":"user","content":user_msg}]

def _sys_with_rollup(s:SessionState)->Dict[str,str]:
    base=s.history[0]["content"]
    if s.rollup: base+="\nSESSION_SUMMARY:"+s.rollup
    return {"role":"system","content":base}

@app.post("/chat")
async def chat(request:Request):
    body=await request.json()
    msg=(body.get("message") or "").strip()
    base=Response()
    sid,s=get_or_create_sid(request,base)
    try_set_faith(msg,s)
    s.history.append({"role":"user","content":msg})
    hits=[]
    if any(w in msg.lower() for w in ["verse","scripture","psalm","quote","ayah","talmud","bible","quran"]):
        hits=hybrid_search(msg,detect_corpus(msg,s))
    reply=call_openai(build_messages(s,msg,hits),stream=False)
    s.history.append({"role":"assistant","content":reply})
    return JSONResponse({"response":reply,"sid":sid,"sources":[{"ref":clean_citation(h),"source":h.get("source","")} for h in hits[:1]]})
