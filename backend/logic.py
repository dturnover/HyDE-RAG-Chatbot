# logic.py
import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import config
import rag

# The OpenAI client is needed here for the query rewrite
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
    """Calculates the Levenshtein edit distance between two strings."""
    if len(s1) > len(s2): s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        new_distances = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2: new_distances.append(distances[i1])
            else: new_distances.append(1 + min((distances[i1], distances[i1 + 1], new_distances[-1])))
        distances = new_distances
    return distances[-1]

SYSTEM_BASE_FLOW = """You are the Fight Chaplain, a professional, empathetic, and strictly non-denominational spiritual guide. Your purpose is to support a person navigating stressful situations by weaving in relevant scripture when their faith is known."""

def system_message(s: SessionState, quote_allowed: bool, retrieval_ctx: Optional[str]) -> Dict[str, str]:
    rag_instruction = ""
    if quote_allowed:
        rag_instruction = "A relevant scripture has been provided in the context below. You MUST seamlessly weave a short, direct quote from this passage's 'text' into your response. Your quote must be enclosed in quotation marks. After the quote, you MUST cite it using an em dash (e.g., — Isaiah 41:10)."
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
            stream=False
        )
        rewritten = completion.choices[0].message.content
        return rewritten if rewritten else user_message
    except Exception:
        return user_message

def get_rag_context(msg: str, s: SessionState) -> Optional[str]:
    should_retrieve = wants_retrieval(msg)
    faith_is_set = bool(s.faith)

    if should_retrieve and faith_is_set:
        rewritten_query = _get_rewritten_query(msg)
        hits = rag.hybrid_search(rewritten_query, s.faith)
        if not hits: return None

        # --- Data Quality Checks ---
        # 1. Try to find the first good hit that has substantial text.
        good_hit = None
        for hit in hits:
            text = hit.get('text', '').strip()
            # Ensure the text is not just a title/summary (at least 5 words)
            if len(text.split()) >= 5:
                good_hit = hit
                break
        
        # If no good hits were found, abort.
        if not good_hit:
            return None

        # 2. Clean up the reference from the good hit.
        ref = good_hit.get('ref', 'Unknown Reference').strip()
        ref = re.sub(r'(\d+)$', r'', ref) # Turns "Luke 12:157" into "Luke 12:15"
        text = good_hit.get('text', '').strip()

        return f"RETRIEVED PASSAGE:\n- Reference: {ref}\n- Text: \"{text}\""
    
    return None

def update_session_metrics(msg: str, s: SessionState) -> None:
    # Placeholder for future logic, like escalation
    pass

def apply_referral_footer(text: str, s: SessionState) -> str:
    # Placeholder for future logic
    return text