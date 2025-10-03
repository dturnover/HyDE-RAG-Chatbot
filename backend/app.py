# Fight Chaplain Backend - app.py
#
# This version implements on-disk streaming RAG to conserve RAM, addressing OOM issues.
# It maintains session state, streaming responses (SSE), and the strict 7-step chaplain flow.

import os, re, json, uuid, math, asyncio, heapq
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import MutableHeaders
from starlette.middleware.base import BaseHTTPMiddleware
from http.cookies import SimpleCookie

# --- 1. CONFIGURATION & EXTERNAL CLIENTS ---

try:
    # Requires 'openai' library
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False
    OpenAI = None  # type: ignore

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")

def oa_client() -> Optional[OpenAI]:
    """Returns an OpenAI client if API key is present."""
    return OpenAI(api_key=OPENAI_API_KEY) if (_HAS_OPENAI and OPENAI_API_KEY) else None

def embed_query(text: str) -> Optional[List[float]]:
    """Generates an embedding for a query using OpenAI (vector search step)."""
    cli = oa_client()
    if not cli: return None
    try:
        r = cli.embeddings.create(model=EMBED_MODEL, input=text)
        return r.data[0].embedding
    except Exception:
        # Logging here would be useful, but we simply return None to disable vector search
        return None

def call_openai(messages: List[Dict[str,str]], stream: bool):
    """Calls OpenAI chat completion, handling dev mode and streaming."""
    cli = oa_client()
    if not cli:
        dev_msg = "⚠️ OpenAI disabled on server; running in dev mode."
        if stream:
            return (w + " " for w in dev_msg.split())
        return dev_msg
    
    kwargs = dict(model=OPENAI_MODEL, messages=messages, temperature=0.2, stream=stream)
    
    try:
        resp = cli.chat.completions.create(**kwargs)
    except Exception as e:
        err_msg = f"OpenAI API Error: {str(e)}"
        if stream:
            return (w + " " for w in err_msg.split())
        return err_msg

    if stream:
        def _gen():
            for ev in resp:
                piece = ev.choices[0].delta.content or ""
                if piece: yield piece
        return _gen()
    
    return resp.choices[0].message.content or ""

# --- 2. SESSION MANAGEMENT & STATE ---

@dataclass
class _ChaplainState:
    """Internal state for the chaplain's escalation logic."""
    turns: int = 0
    distress_hits: int = 0
    crisis_hits: int = 0
    escalate: str = "none" # "none" | "watch" | "refer-faith" | "refer-mental-health"
    faith_pref: str = ""

@dataclass
class SessionState:
    """Represents the state of a user session."""
    # Note: history[0] is always the ONBOARD_GREETING substitute when we save.
    history: List[Dict[str,str]] = field(default_factory=lambda: [{"role":"system","content":""}])
    rollup: Optional[str] = None # Placeholder for future memory summarization
    faith: Optional[str] = None  # 'bible_asv' | 'quran' | etc.
    _chap: _ChaplainState = field(default_factory=_ChaplainState)

SESSIONS: Dict[str, SessionState] = {}

def get_or_create_sid(request: Request, response: Optional[Response] = None) -> Tuple[str, SessionState]:
    """Gets/creates session ID and state, setting a cookie if response is provided."""
    sid = request.headers.get("X-Session-Id") or request.cookies.get("sid")
    is_new_session = not sid
    
    if not sid:
        sid = uuid.uuid4().hex

    if sid not in SESSIONS:
        # FIX: When a session is created, the first item in history is the system placeholder.
        # This fixes a bug where history was being trimmed incorrectly when empty.
        new_session = SessionState()
        SESSIONS[sid] = new_session
        is_new_session = True
        
    # If a Response object is passed (meaning this is the first interaction), set the cookie.
    if response and is_new_session:
        # Secure, httponly, and samesite="none" are crucial for cross-site cookie settings
        response.set_cookie("sid", sid, httponly=True, samesite="none", secure=True, path="/")
        
    return sid, SESSIONS[sid]

# --- 3. INDEX DISCOVERY (ON-DISK STREAMING SETUP) ---
# NOTE: This section replaces the old memory-loading index logic.

SOURCE_NORMALIZER = {
    "asvhb": "bible_asv", "asv": "bible_asv", "nrsv": "bible_nrsv",
    "tanakh": "tanakh", "quran": "quran", "gita": "gita", "bhagavad gita": "gita",
    "dhammapada": "dhammapada", "bible_asv": "bible_asv", "bible_nrsv": "bible_nrsv",
}
VALID_CORPORA = set(SOURCE_NORMALIZER.values())

def _candidate_index_dirs() -> List[Path]:
    """Determines potential index directories for RAG files."""
    here = Path(__file__).parent.resolve()
    env_dir = Path(os.getenv("RAG_DIR", "").strip())
    
    cands = [p for p in [env_dir, here / "indexes", here.parent / "indexes", Path("/opt/render/project/src/indexes")] if p and p.is_dir()]
    return list(dict.fromkeys(cands))

def discover_corpora() -> Dict[str, Path]:
    """Locates the on-disk path for each corpus JSONL file (does NOT load content)."""
    names = {
        "bible_asv": "bible_asv_embed.jsonl",
        "bible_nrsv": "bible_nrsv_embed.jsonl",
        "quran":     "quran_embed.jsonl",
        "tanakh":    "tanakh_embed.jsonl",
        "gita":      "gita_embed.jsonl",
        "dhammapada":"dhammapada_embed.jsonl",
    }
    legacy_names = {k: f"{k}.jsonl" for k in VALID_CORPORA}

    found: Dict[str, Path] = {}
    
    for base in _candidate_index_dirs():
        # Search for modern names first
        for key, fname in names.items():
            p = base / fname
            if key not in found and p.exists() and p.stat().st_size > 0:
                found[key] = p
        # Search for legacy names second
        for key, fname in legacy_names.items():
            if key not in found:
                p = base / fname
                if p.exists() and p.stat().st_size > 0:
                    found[key] = p
                    
    return found

# Global dictionary holding corpus name -> file path. This is tiny.
CORPUS_FILES: Dict[str, Path] = discover_corpora()

def corpus_counts() -> Dict[str, int]:
    """Lazy line counts for health/diag endpoint."""
    _cache = getattr(corpus_counts, "_cache", None)
    if isinstance(_cache, dict): return _cache
    counts: Dict[str, int] = {}
    for k, path in CORPUS_FILES.items():
        try:
            # Simple line count to estimate size without loading
            n = 0
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for _ in f: n += 1
            counts[k] = n
        except Exception:
            counts[k] = 0
    setattr(corpus_counts, "_cache", counts)
    return counts

# --- 4. TEXT & MATH UTILITIES (HYBRID SEARCH FOUNDATION) ---

def tokenize(s: str) -> set[str]: 
    """Tokenizes text for lexical search."""
    return set(re.findall(r"[A-Za-z0-9]+", s.lower()))

def jaccard(a: set[str], b: set[str]) -> float:
    """Calculates Jaccard similarity (keyword overlap)."""
    if not a or not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return inter / union if union else 0.0

def cos(a: List[float], b: List[float]) -> float:
    """Calculates cosine similarity (vector angle)."""
    if not (a and b): return 0.0
    s, na, nb = 0.0, 0.0, 0.0
    for x, y in zip(a, b):
        s += x * y; na += x * x; nb += y * y
    return s / math.sqrt(na * nb) if na > 0 and nb > 0 else 0.0

@dataclass
class _Stat:
    """Welford's algorithm for streaming mean/variance calculation (low RAM)."""
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0
    def push(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)    
    @property
    def std(self) -> float:
        # Use 1.0 as fallback to prevent division by zero and maintain z-score scale
        return math.sqrt(self.M2 / self.n) if self.n >= 2 else 1.0 


def _iter_jsonl_rows(path: Path):
    """Generator to stream rows from a JSONL file on disk."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def _normalize_output_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensures RAG results match the expected structure for the system prompt."""
    out_rows = []
    for row in rows:
        out_rows.append({
            "id": row.get("id") or row.get("ref") or "",
            "ref": row.get("ref") or row.get("book") or row.get("id") or "",
            "source": row.get("source") or "",
            "book": row.get("book") or "",
            "chapter": row.get("chapter") or row.get("chap") or "",
            "verse": row.get("verse") or row.get("ayah") or "",
            "text": row.get("text") or "",
            "embedding": row.get("embedding") or [],
        })
    return out_rows

def _edit_distance(s1, s2):
    """Simple, non-library Levenshtein implementation for typo checking."""
    # Source: https://en.wikipedia.org/wiki/Levenshtein_distance#Iterative_with_full_matrix
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        new_distances = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                new_distances.append(distances[i1])
            else:
                new_distances.append(1 + min((distances[i1], distances[i1+1], new_distances[-1])))
        distances = new_distances
    return distances[-1]


def hybrid_search(query: str, corpus_name: str, top_k: int = 6) -> List[Dict[str, Any]]:
    """
    On-disk streaming Hybrid (Jaccard + Cosine) search with Dual-Search Fallback.
    Consolidates candidates from both lexical and vector searches for a robust blend.
    """
    path = CORPUS_FILES.get(corpus_name)
    if not path or not path.exists():
        return []

    q_tokens = tokenize(query)
    q_emb = embed_query(query)
    
    OVER = max(top_k * 8, 50) # Over-fetch window size
    
    # --- Data Structures to Store Results from Two Passes ---
    all_candidates: Dict[int, Dict[str, Any]] = {} # Master collection of all potential results
    lex_scores: Dict[int, float] = {} # Jaccard score
    vec_scores: Dict[int, float] = {} # Cosine score
    
    # --- Statistics for Normalization ---
    lex_stat = _Stat()
    vec_stat = _Stat()
    
    # --- Top Candidate Heaps (used for final pool selection) ---
    lex_heap: list[tuple[float, int]] = [] # (Jaccard, index)
    vec_heap: list[tuple[float, int]] = [] # (Cosine, index)
    
    # --- Streaming Pass (Consolidated) ---
    for i, row in enumerate(_iter_jsonl_rows(path)):
        # 1. Lexical Scoring (Jaccard)
        txt = row.get("text") or ""
        js = jaccard(q_tokens, tokenize(txt))
        lex_stat.push(js)
        
        if js > 0.0:
            lex_scores[i] = js
            all_candidates[i] = row
            if len(lex_heap) < OVER:
                heapq.heappush(lex_heap, (js, i))
            elif js > lex_heap[0][0]:
                heapq.heapreplace(lex_heap, (js, i))
                
        # 2. Vector Scoring (Cosine)
        e = row.get("embedding")
        if isinstance(e, list) and q_emb:
            s = cos(q_emb, e)
            vec_stat.push(s)
            
            vec_scores[i] = s
            all_candidates[i] = row # Add to master if not already added by lexical
            
            if len(vec_heap) < OVER:
                heapq.heappush(vec_heap, (s, i))
            elif s > vec_heap[0][0]:
                heapq.heapreplace(vec_heap, (s, i))

    # --- Consolidation and Normalization ---
    
    # Identify unique indices selected by either lexical or vector search (the pool to score)
    top_indices = set(idx for _, idx in lex_heap) | set(idx for _, idx in vec_heap)
    
    if not top_indices:
        return [] # Both searches failed entirely
    
    # Global stats for Z-scoring
    mean_L, std_L = lex_stat.mean, (lex_stat.std or 1.0)
    mean_V, std_V = vec_stat.mean, (vec_stat.std or 1.0)
    
    # 1. Calculate Z-scores for the consolidated pool
    Lz: Dict[int, float] = {}
    Vz: Dict[int, float] = {}
    
    for idx in top_indices:
        # Use mean if score is missing (meaning it wasn't in the top OVER candidates of that type)
        js = lex_scores.get(idx, mean_L) 
        cs = vec_scores.get(idx, mean_V)
        
        # Calculate Z-scores:
        Lz[idx] = (js - mean_L) / std_L
        Vz[idx] = (cs - mean_V) / std_V

    # --- BLEND ---
    # Alpha was 0.8 (80% lexical). Reduced to 0.5 (50% lexical, 50% vector) to give
    # equal weight to conceptual (vector) search, which is necessary for broad distress terms.
    alpha = 0.5 
    blended: List[tuple[float, int, Dict[str, Any]]] = []
    
    for idx in top_indices:
        lz = Lz[idx]
        vz = Vz[idx]
        sc = alpha*lz + (1 - alpha)*vz
        blended.append((sc, idx, all_candidates[idx]))
        
    blended.sort(reverse=True, key=lambda t: t[0])
    
    # Return the top K candidates
    return _normalize_output_rows([r for _, _, r in blended[:top_k]])

# --- 5. CHAPLAIN AI LOGIC & STATE UPDATE ---

ONBOARD_GREETING = "I am the Fight Chaplain. My role is to offer spiritual guidance and connect you with verses from your tradition. How can I support you today? (Please specify your faith, e.g., 'Christian,' 'Muslim,' or 'Hindu.')"

SYSTEM_BASE_FLOW = """
You are the Fight Chaplain. You are a highly professional, empathetic, and strictly non-denominational spiritual guide.
Your entire purpose is to support a person navigating stressful situations or seeking spiritual insight.
You must adhere to the following 7-step flow in every response (keep responses **concise and direct**):
1. **Acknowledge and Validate:** Start by recognizing the user's emotion or need with empathy.
2. **Determine RAG:** Check if a quote is provided in the context.
3. **If RAG is available (Quote Allowed):** Weave the provided scripture, quote, or passage *seamlessly* and briefly into your response. Do not quote more than one short line verbatim. The entire reply must be centered on the retrieved text.
4. **If RAG is NOT available (No Quote Allowed):** DO NOT under any circumstances invent or quote scripture, verses, or religious passages. Instead, follow the specific RAG RULE below.
5. **Guidance and Context:** Provide brief, supportive, and action-oriented guidance based on the context.
6. **Referral Check:** If the session status demands it, append the mandatory referral message (either faith leader or mental health).
7. **Invite Continuation:** End with an open-ended question to encourage the user to continue the conversation.
"""

# Keywords for RAG trigger and escalation
ASK_WORDS = {"verse","scripture","psalm","quote","passage","ayah","surah","quran","bible","tanakh","gita","dhammapada"}
DISTRESS_KEYWORDS = {"scared","anxious","worried","hurt","down","lost","depressed","angry","grief","lonely","alone","doubt","stress", "nervous", "afraid"} # ADDED "nervous" and "afraid"
CRISIS_KEYWORDS = {"panic","suicide","kill myself","hopeless","end it","emergency","self-harm"}

# Keywords to detect faith preference
FAITH_KEYWORDS = {
    "catholic": "bible_nrsv", "orthodox": "bible_nrsv", "protestant": "bible_asv", 
    "evangelical": "bible_asv", "christian": "bible_asv", "jewish": "tanakh", 
    "jew": "tanakh", "hebrew": "tanakh", "muslim": "quran", "islam": "quran", 
    "quran": "quran", "koran": "quran", "hindu": "gita", "bhagavad": "gita", 
    "gita": "gita", "buddhist": "dhammapada", "buddhism": "dhammapada", 
    "dhammapada": "dhammapada",
}

# Very big corpora (avoid scanning unless faith is known)
BIG_CORPORA = {"bible_nrsv", "tanakh", "bible_asv"}

def try_set_faith(msg: str, s: SessionState) -> None:
    """Sets the session faith preference if a keyword is found (now includes typo check)."""
    m = msg.lower()
    m_tokens = tokenize(msg)
    
    for k, corp in FAITH_KEYWORDS.items():
        # 1. Check for perfect match (fastest)
        if f" {k} " in f" {m} " or m.startswith(k) or m.endswith(k):
            s.faith = corp
            return
            
        # 2. Check for close match on individual tokens (typo handling)
        for token in m_tokens:
            # Only check if token is close to a faith keyword length for efficiency
            if abs(len(token) - len(k)) <= 2: 
                if _edit_distance(token, k) <= 1: # Allow 1 typo (e.g., cathulic -> catholic)
                    s.faith = corp
                    return

def detect_corpus(msg: str, s: SessionState) -> str:
    """Determines the most appropriate corpus based on session state or message keywords."""
    # FIX: If s.faith is set, it overrides keyword detection for the current turn.
    if s.faith: return s.faith 
    
    m = msg.lower()
    for k, corp in FAITH_KEYWORDS.items():
        if k in m: return corp
        
    if any(w in m for w in ["surah","ayah"]): return "quran"
    if any(w in m for w in ["tractate","mishnah","gemara"]): return "tanakh"
    if "bible" in m: return "bible_asv"
    return "bible_asv" # Default if nothing is detected

def allowed_corpus_for(msg: str, s: SessionState) -> Optional[str]:
    """
    Guard function: Chooses a corpus but prevents scanning giant files 
    if the user's faith is not yet established in the session state.
    """
    corp = detect_corpus(msg, s)
    
    # FIX: If faith is set in the session, RAG is ALWAYS allowed for that corpus.
    if s.faith: 
        return corp
    
    # If no faith set, we fall back to the original guard logic:
    # Deny RAG if a large corpus is implied but faith is not confirmed by session state.
    if corp in BIG_CORPORA:
        m = msg.lower()
        if any(w in m for w in ["qur", "koran", "allah"]): return "quran"
        if "gita" in m: return "gita"
        if "dhamma" in m or "buddh" in m: return "dhammapada"
        
        # Deny RAG if large corpus is implied but faith is not confirmed
        return None
        
    return corp

def wants_retrieval(msg: str) -> bool:
    """Checks if the user's message indicates a need for RAG."""
    m = f" {msg.lower()} "
    return any(f" {w} " in m for w in ASK_WORDS | DISTRESS_KEYWORDS)

def clean_passage(text: str) -> str:
    """Cleans up text for the system prompt to maintain quality."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:200] + '...' if len(text) > 200 else text

def natural_ref(hit: Dict[str, Any]) -> str:
    """Formats the reference for the system prompt."""
    source = hit.get("source", "").upper()
    book = hit.get("book", "")
    chap = hit.get("chapter", "")
    verse = hit.get("verse", "")
    if source == "QURAN":
        return f"Qur'an (Surah {chap}:{verse})"
    if source in ("BIBLE_ASV", "BIBLE_NRSV", "TANAKH"):
        return f"{book} {chap}:{verse}"
    if source == "GITA":
        return f"Bhagavad Gita ({chap}.{verse})"
    if source == "DHAMMAPADA":
        return f"Dhammapada ({chap}.{verse})"
    return f"{book} {chap} {verse} ({source})"

def update_session_metrics(msg: str, s: SessionState) -> None:
    """Updates turn count and escalation metrics."""
    s._chap.turns += 1
    m = msg.lower()
    
    # Check for crisis keywords first
    if any(w in m for w in CRISIS_KEYWORDS):
        s._chap.crisis_hits += 1
        s._chap.escalate = "refer-mental-health"
        return

    # Check for general distress keywords
    if any(w in m for w in DISTRESS_KEYWORDS):
        s._chap.distress_hits += 1
        
    # Escalation rules
    if s._chap.crisis_hits > 0 and s._chap.escalate != "refer-mental-health":
         s._chap.escalate = "refer-mental-health"
    elif s._chap.turns >= 12 and s._chap.escalate == "none":
        s._chap.escalate = "refer-faith" # Long conversation, suggest a faith leader
    elif s._chap.distress_hits >= 5 and s._chap.escalate == "none":
        s._chap.escalate = "refer-faith" # Repeated distress, suggest a faith leader
    elif s._chap.escalate == "watch" and s._chap.distress_hits == 0:
        s._chap.escalate = "none" # De-escalate if distress subsides

def apply_referral_footer(text: str, status: str) -> str:
    """Appends mandatory referral text based on escalation status."""
    if status == "refer-mental-health":
        text += "\n\n**Mandatory Referral:** If you or someone you know is in crisis, please call or text 988 in the US/Canada, or search for a local crisis line immediately. Your safety is paramount."
    elif status == "refer-faith":
        text += "\n\n**Note:** For deeper, personalized guidance, please consider reaching out to a local faith leader or spiritual counselor who can provide direct support and prayer."
    return text

def system_message(s: SessionState, quote_allowed: bool, faith_known: bool, retrieval_ctx: Optional[str]) -> Dict[str, str]:
    """Constructs the LLM system message with all context and rules."""
    
    rag_instruction = "You are FORBIDDEN from inventing or quoting scripture."
    
    if quote_allowed:
        # RAG succeeded, use the quote
        rag_instruction = "You MUST use the provided passage and adhere to step 3."
    else: # No quote is provided by RAG
        if faith_known:
            # FIX: Faith is known but RAG failed to retrieve a quote (e.g., poor vector search results).
            # The LLM must acknowledge the known faith and provide practical guidance.
            rag_instruction += (
                f" Your session faith is set to **{s.faith}**. DO NOT claim the user's faith is unknown. "
                "Since no quote was retrieved this turn, provide only empathetic, practical, and non-denominational guidance (Step 5)."
            )
        else:
            # FIX: Faith is unknown, follow the original instruction.
            rag_instruction += (
                " The user's faith is UNKNOWN. You must gently ask the user to specify their faith/tradition to unlock scripture support."
            )
    
    session_status = (
        f"SESSION STATUS: Escalation={s._chap.escalate}. Turns={s._chap.turns}. "
        f"Faith set={s.faith or 'None'}. Quote is allowed={quote_allowed}."
    )
    
    full_prompt = (
        SYSTEM_BASE_FLOW + "\n"
        "--- CONTEXT ---\n"
        f"SESSION HISTORY ROLLUP (If available, summarize user intent): {s.rollup or 'N/A'}\n"
        f"CURRENT SESSION STATUS: {session_status}\n"
        f"RAG RULE: {rag_instruction}\n"
        f"{retrieval_ctx or 'No passages retrieved this turn. The LLM must not quote.'}\n"
    )

    return {"role": "system", "content": full_prompt}

# --- 6. SHARED CHAT LOGIC (DRY principle) ---

def handle_chat_turn(s: SessionState, msg: str) -> Tuple[List[Dict[str,str]], Optional[Dict[str,str]]]:
    """
    Handles common chat logic: history update, RAG/System message construction.
    Returns (messages_for_llm, source_info_for_response)
    """
    # 1. Update Session and History
    try_set_faith(msg, s)
    update_session_metrics(msg, s)
    
    # Append user message for the LLM call
    s.history.append({"role":"user","content":msg})
    
    # Simple history trimming to keep the context window manageable
    # We keep the system prompt + last 39 messages
    if len(s.history) > 40:
        s.history = s.history[:1] + s.history[-39:]
        
    messages = s.history[1:] # Exclude the placeholder system message

    # 2. RAG Check (Step 2)
    quote_allowed, retrieval_ctx, source_info = False, None, None

    if wants_retrieval(msg):
        # NEW LOGIC: Check if RAG is allowed given the memory constraint
        corp = allowed_corpus_for(msg, s) 
        
        if corp and corp in CORPUS_FILES: # Only run RAG if a corpus is allowed and found
            # FIX: Added a check here to ensure the corpus is actually available
            if CORPUS_FILES[corp].exists() and CORPUS_FILES[corp].stat().st_size > 0:
                hits = hybrid_search(msg, corp, top_k=6)
            else:
                hits = [] # File not found or empty
        else:
            hits = [] # RAG denied by guard or corp is None

        if hits:
            top = hits[0]
            clean_text = clean_passage(top.get("text",""))
            pretty_ref = natural_ref(top)
            
            retrieval_ctx = (
                "RETRIEVED PASSAGES (you may quote EXACTLY ONE short line verbatim; then weave brief guidance around it):\n"
                f"- {pretty_ref} :: {clean_text}"
            )
            quote_allowed = True
            source_info = {"ref": pretty_ref, "source": top.get("source","")}

    # 3. Build System Message
    sys_msg = system_message(s, quote_allowed=quote_allowed, faith_known=bool(s.faith), retrieval_ctx=retrieval_ctx)
    
    # Prepend the dynamic system message
    messages = [sys_msg] + messages
    
    return messages, source_info

# --- 7. FASTAPI APPLICATION SETUP ---

app = FastAPI(title="Fight Chaplain Backend")

# Custom Middleware for CORS and Cookie Handling
class CORSCookieMiddleware(BaseHTTPMiddleware):
    """Handles setting necessary CORS headers and ensures cookies are available."""
    async def dispatch(self, request: Request, call_next):
        # --- Preflight/Response setup ---
        response = await call_next(request)
        
        origin = request.headers.get("origin")
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
        else:
             response.headers["Access-Control-Allow-Origin"] = "*" # Fallback for non-browser clients

        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Session-Id"
        response.headers["Vary"] = "Origin" # Essential when dynamically setting CORS Origin

        # --- Initial Cookie Setting (if it's a new session) ---
        if 'sid' not in request.cookies:
            # We need a temporary response to set the cookie before returning
            temp_response = Response()
            # FIX: We only call get_or_create_sid here to ensure cookie setting logic runs once.
            sid, session_state = get_or_create_sid(request, temp_response)
            
            # Transfer cookies from the temp response to the final response
            if 'set-cookie' in temp_response.headers:
                if 'set-cookie' in response.headers:
                    # Append new cookie header to existing ones if present
                    response.headers.append('set-cookie', temp_response.headers['set-cookie'])
                else:
                    response.headers['set-cookie'] = temp_response.headers['set-cookie']
                    
        return response

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(CORSCookieMiddleware) # Use custom middleware for cookie/CORS interaction

@app.get("/diag_rag")
async def diag_rag():
    """Diagnostic endpoint to check corpus file availability and size."""
    return {
        "status": "ok",
        "has_openai": _HAS_OPENAI,
        "openai_model": OPENAI_MODEL,
        "rag_dir_candidates": [str(p) for p in _candidate_index_dirs()],
        "available_corpora": {k: str(v) for k, v in CORPUS_FILES.items()},
        "corpus_line_counts": corpus_counts(),
        "memory_model": "On-Disk Streaming RAG (Low RAM usage)",
    }


# --- 8. CHAT ROUTES ---

@app.post("/chat")
async def chat_non_streaming(request: Request, response: Response):
    """Standard POST route for non-streaming chat responses."""
    
    sid, s = get_or_create_sid(request, response)
    
    try:
        data = await request.json()
        msg = data.get("message", "").strip()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON format")

    if not msg:
        return JSONResponse({"message": "No message provided.", "sources": []})

    messages, source_info = handle_chat_turn(s, msg)
    
    # Non-streaming call
    assistant_out = call_openai(messages, stream=False)
    
    # 5. Final Step: Apply referral and save history
    final_response = apply_referral_footer(assistant_out, s._chap.escalate)
    s.history.append({"role": "assistant", "content": final_response})
    
    return JSONResponse({
        "message": final_response,
        "sources": source_info,
        "session_id": sid,
        "faith": s.faith,
    })

@app.get("/chat_sse")
async def chat_streaming(request: Request, response: Response):
    """GET route for Server-Sent Events (SSE) streaming chat responses."""
    
    # 1. Get Session & Handle Setup
    # FIX: Ensure cookies are read correctly by passing the response object here, 
    # even though we don't use the returned response object (it just ensures cookie setting).
    sid, s = get_or_create_sid(request, response) 
    msg = request.query_params.get("message", "").strip()

    if not msg:
        # Return a simple event stream signaling no content
        return StreamingResponse(
            content="event: error\ndata: No message provided\n\n",
            media_type="text/event-stream"
        )
        
    messages, source_info = handle_chat_turn(s, msg)

    # 2. Asynchronously Generate Stream
    async def agen():
        assistant_out = ""
        last_yield = 0
        chunk_min_size = 8
        
        # 2a. Send initial session/source information (event: start)
        yield f"event: start\ndata: {json.dumps({'session_id': sid, 'faith': s.faith or 'none'})}\n\n"

        # 2b. Send source info if RAG was used (event: sources)
        if source_info:
            yield f"event: sources\ndata: {json.dumps(source_info)}\n\n"
            
        # 3. Call OpenAI and Stream Text (event: message)
        llm_stream = call_openai(messages, stream=True)
        if llm_stream:
            for chunk in llm_stream:
                if not isinstance(chunk, str): continue
                
                assistant_out += chunk
                
                # Check for chunking conditions
                if (len(assistant_out) - last_yield >= chunk_min_size) or ('\n' in assistant_out[last_yield:]):
                    # Yield the new text chunk
                    new_chunk = assistant_out[last_yield:]
                    # FIX: Use JSON dumps to properly escape newlines and quotes in the chunk
                    yield f"event: message\ndata: {json.dumps({'text': new_chunk})}\n\n"
                    last_yield = len(assistant_out)

                await asyncio.sleep(0.001) # Small pause to allow context switching
        
        # 4. Final text push (in case the last chunk was too small)
        if len(assistant_out) > last_yield:
             new_chunk = assistant_out[last_yield:]
             yield f"event: message\ndata: {json.dumps({'text': new_chunk})}\n\n"
        
        # 5. Final Step: Apply referral footer (event: footer)
        final_response = apply_referral_footer(assistant_out, s._chap.escalate)
        footer_text = final_response[len(assistant_out):] # Only the added referral text
        
        if footer_text.strip():
            yield f"event: footer\ndata: {json.dumps({'text': footer_text})}\n\n"
        
        # 6. Save final response to history
        s.history.append({"role": "assistant", "content": final_response})
        
        # 7. Close stream
        yield "event: end\ndata: {}\n\n"

    # Set appropriate headers for SSE
    return StreamingResponse(agen(), media_type="text/event-stream")
