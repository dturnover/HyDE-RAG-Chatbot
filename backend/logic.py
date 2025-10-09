# logic.py
import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import config
import rag

# Note: The OpenAI client is now needed here for the query rewrite
client = rag.client

@dataclass
class SessionState:
    """A temporary object to hold state for a single request."""
    history: List[Dict[str, str]] = field(default_factory=list)
    faith: Optional[str] = None
    
    @property
    def _chap(self):
        class MockChap:
            escalate = "none"
            turns = len(self.history) // 2
        return MockChap()

def _edit_distance(s1: str, s2: str) -> int:
    if len(s1) > len(s2): s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        new_distances = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2: new_distances.append(distances[i1])
            else: new_distances.append(1 + min((distances[i1], distances[i1 + 1], new_distances[-1])))
        distances = new_distances
    return distances[-1]

SYSTEM_BASE_FLOW = """You are the Fight Chaplain...""" # Omitted for brevity

def system_message(s: SessionState, quote_allowed: bool, retrieval_ctx: Optional[str]) -> Dict[str, str]:
    rag_instruction = ""
    if quote_allowed:
        rag_instruction = "A relevant scripture has been provided in the context. You MUST seamlessly weave a short, direct quote from this passage into your response, followed by an em dash and the citation (e.g., — Isaiah 41:10)."
    elif s.faith:
        rag_instruction = f"The user's faith is known ({s.faith}), but no scripture was retrieved for this turn. Provide an empathetic, practical response without inventing or mentioning scripture."
    else:
        rag_instruction = "The user's faith is UNKNOWN. Do not provide scripture. Gently ask them to share their faith tradition if they are seeking scriptural support."
    session_status = f"SESSION STATUS: Faith set={s.faith or 'None'}. Quote is allowed={quote_allowed}."
    full_prompt = (f"{SYSTEM_BASE_FLOW}\n--- CONTEXT ---\n"
                   f"CURRENT SESSION STATUS: {session_status}\n"
                   f"RAG RULE: {rag_instruction}\n"
                   f"{retrieval_ctx or 'No passages retrieved.'}\n")
    return {"role": "system", "content": full_prompt}

def try_set_faith(msg: str, s: SessionState) -> None:
    if s.faith: return
    m = msg.lower()
    for keyword, faith_id in config.FAITH_KEYWORDS.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', m):
            s.faith = faith_id
            return
    msg_tokens = set(re.findall(r'\w+', m))
    for keyword, faith_id in config.FAITH_KEYWORDS.items():
        for token in msg_tokens:
            if abs(len(token) - len(keyword)) <= 2 and _edit_distance(token, keyword) <= 1:
                s.faith = faith_id
                return

def wants_retrieval(msg: str) -> bool:
    m_lower = msg.lower()
    return any(word in m_lower for word in config.ASK_WORDS | config.DISTRESS_KEYWORDS)

def _get_rewritten_query(user_message: str) -> str:
    """Uses a fast LLM call to rewrite the user's query for better RAG results."""
    if not client: return user_message
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a search query generation expert. Rewrite the user's message into a concise, high-quality search query for finding relevant passages in a religious text database."},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
            stream=False # We need the full response, not a stream
        )
        rewritten = completion.choices[0].message.content
        return rewritten if rewritten else user_message
    except Exception:
        # If the rewrite fails for any reason, fall back to the original message
        return user_message

def get_rag_context(msg: str, s: SessionState) -> Optional[str]:
    should_retrieve = wants_retrieval(msg)
    faith_is_set = bool(s.faith)

    if should_retrieve and faith_is_set:
        # ★★★ LLM QUERY TRANSFORMATION ★★★
        # First, rewrite the user's query into a better one.
        rewritten_query = _get_rewritten_query(msg)
        
        # Then, use that rewritten query to perform the search.
        hits = rag.hybrid_search(rewritten_query, s.faith)
        
        if not hits:
            return None

        top_hit = hits[0]
        ref = top_hit.get('ref') or f"{top_hit.get('book', '')} {top_hit.get('chapter', '')}:{top_hit.get('verse', '')}".strip()
        text = re.sub(r'\s+', ' ', top_hit.get('text', '')).strip()
        return f"RETRIEVED PASSAGE (weave this in seamlessly):\n- {ref} :: {text}"
    
    return None