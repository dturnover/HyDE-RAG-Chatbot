# logic.py
import re
from typing import Dict, Optional
from state import SessionState
import config
import rag

# --- ★★★ NEW: Typo Detection Helper Function ★★★ ---
def _edit_distance(s1: str, s2: str) -> int:
    """Calculates the Levenshtein edit distance between two strings."""
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        new_distances = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                new_distances.append(distances[i1])
            else:
                new_distances.append(1 + min((distances[i1], distances[i1 + 1], new_distances[-1])))
        distances = new_distances
    return distances[-1]

# --- Prompt Templates ---
SYSTEM_BASE_FLOW = """
You are the Fight Chaplain, a professional, empathetic, and strictly non-denominational spiritual guide.
Your purpose is to support a person navigating stressful situations. Adhere to this 7-step flow:
1. Acknowledge and Validate: Recognize the user's emotion with empathy.
2. Determine RAG: Check if a quote is provided in the context.
3. If RAG is available: Weave the provided scripture seamlessly and briefly into your response.
4. If RAG is NOT available: DO NOT invent scripture. Follow the specific RAG RULE below.
5. Guidance and Context: Provide brief, supportive, action-oriented guidance.
6. Referral Check: If the session status demands it, append the mandatory referral message.
7. Invite Continuation: End with an open-ended question.
"""

def system_message(s: SessionState, quote_allowed: bool, retrieval_ctx: Optional[str]) -> Dict[str, str]:
    """Constructs the full system prompt for the LLM."""
    rag_instruction = ""
    if quote_allowed:
        rag_instruction = "You MUST use the provided passage and adhere to step 3."
    elif s.faith:
        rag_instruction = (
            f"You are FORBIDDEN from inventing scripture. Your session faith is set to **{s.faith}**. "
            "Since no quote was retrieved, provide only empathetic, practical, and non-denominational guidance."
        )
    else:
        rag_instruction = (
            "You are FORBIDDEN from inventing scripture. The user's faith is UNKNOWN. "
            "You must gently ask the user to specify their faith/tradition to unlock scripture support."
        )
    session_status = (
        f"SESSION STATUS: Escalation={s._chap.escalate}. Turns={s._chap.turns}. "
        f"Faith set={s.faith or 'None'}. Quote is allowed={quote_allowed}."
    )
    full_prompt = (
        f"{SYSTEM_BASE_FLOW}\n--- CONTEXT ---\n"
        f"CURRENT SESSION STATUS: {session_status}\n"
        f"RAG RULE: {rag_instruction}\n"
        f"{retrieval_ctx or 'No passages retrieved. The LLM must not quote.'}\n"
    )
    return {"role": "system", "content": full_prompt}

# --- Chaplain Logic Functions ---

def try_set_faith(msg: str, s: SessionState) -> None:
    """Sets the session faith using exact match and then typo checking."""
    m = msg.lower()

    # --- ★★★ UPDATED: Two-Step Faith Detection ★★★ ---

    # Step 1: Look for an exact, whole-word match (fast and precise).
    for keyword, faith_id in config.FAITH_KEYWORDS.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', m):
            s.faith = faith_id
            return

    # Step 2: If no exact match, check for typos (Levenshtein distance).
    msg_tokens = set(re.findall(r'\w+', m))
    for keyword, faith_id in config.FAITH_KEYWORDS.items():
        for token in msg_tokens:
            # Only check for typos on words of similar length for efficiency
            if abs(len(token) - len(keyword)) <= 2:
                if _edit_distance(token, keyword) <= 1: # Allow 1 typo
                    s.faith = faith_id
                    return

def wants_retrieval(msg: str) -> bool:
    """Checks if the user's message indicates a need for RAG."""
    m_lower = msg.lower()
    return any(word in m_lower for word in config.ASK_WORDS | config.DISTRESS_KEYWORDS)

def get_rag_context(msg: str, s: SessionState) -> Optional[str]:
    """Performs RAG and formats the context string for the prompt."""
    if not wants_retrieval(msg):
        return None

    if not s.faith:
        return None

    hits = rag.hybrid_search(msg, s.faith)
    if not hits:
        return None

    top_hit = hits[0]
    ref = top_hit.get('ref', 'Unknown Reference')
    text = re.sub(r'\s+', ' ', top_hit.get('text', '')).strip()
    return f"RETRIEVED PASSAGE (weave this in seamlessly):\n- {ref} :: {text}"

def update_session_metrics(msg: str, s: SessionState) -> None:
    """Updates turn count and escalation metrics."""
    s._chap.turns += 1
    m_lower = msg.lower()

    if any(word in m_lower for word in config.CRISIS_KEYWORDS):
        s._chap.crisis_hits += 1
        s._chap.escalate = "refer-mental-health"
    elif any(word in m_lower for word in config.DISTRESS_KEYWORDS):
        s._chap.distress_hits += 1
    
    if s._chap.distress_hits >= 5 and s._chap.escalate == "none":
        s._chap.escalate = "refer-faith"

def apply_referral_footer(text: str, status: str) -> str:
    """Appends mandatory referral text if needed."""
    if status == "refer-mental-health":
        text += "\n\n**Mandatory Referral:** If you are in crisis, please call or text 988 in the US/Canada, or find a local crisis line. Your safety is most important."
    elif status == "refer-faith":
        text += "\n\n**Note:** For deeper guidance, consider reaching out to a local faith leader or spiritual counselor."
    return text