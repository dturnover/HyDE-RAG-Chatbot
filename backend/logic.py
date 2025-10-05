# logic.py
import re
from typing import Dict, Optional
from state import SessionState
import config
import rag

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

SYSTEM_BASE_FLOW = """
You are the Fight Chaplain...
""" # NOTE: Truncated for brevity, the rest of the file is the same as before.

def system_message(s: SessionState, quote_allowed: bool, retrieval_ctx: Optional[str]) -> Dict[str, str]:
    # This function remains the same
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

# --- ★★★ DEBUG VERSION OF try_set_faith ★★★ ---
def try_set_faith(msg: str, s: SessionState) -> None:
    """Sets the session faith using exact match and then typo checking."""
    print("\n--- [DEBUG] Entering try_set_faith ---")
    print(f"[DEBUG] Received message: '{msg}'")
    m = msg.lower()

    # Step 1: Look for an exact, whole-word match.
    print("[DEBUG] Starting Step 1: Exact Match Check")
    for keyword, faith_id in config.FAITH_KEYWORDS.items():
        print(f"[DEBUG]   - Checking keyword: '{keyword}'")
        match = re.search(r'\b' + re.escape(keyword) + r'\b', m)
        if match:
            print(f"[DEBUG]   ✔✔✔ EXACT MATCH FOUND for '{keyword}'!")
            s.faith = faith_id
            print(f"[DEBUG] Session faith set to: '{s.faith}'. Exiting function.")
            print("--- [DEBUG] Leaving try_set_faith ---\n")
            return
    print("[DEBUG] Step 1 finished. No exact match found.")

    # Step 2: If no exact match, check for typos.
    print("\n[DEBUG] Starting Step 2: Typo Check")
    msg_tokens = set(re.findall(r'\w+', m))
    print(f"[DEBUG] Message tokens: {msg_tokens}")
    for keyword, faith_id in config.FAITH_KEYWORDS.items():
        for token in msg_tokens:
            # Only check for typos on words of similar length
            if abs(len(token) - len(keyword)) <= 2:
                dist = _edit_distance(token, keyword)
                print(f"[DEBUG]   - Comparing token:'{token}' to keyword:'{keyword}'. Distance: {dist}")
                if dist <= 1: # Allow 1 typo
                    print(f"[DEBUG]   ✔✔✔ TYPO MATCH FOUND! ('{token}' is close to '{keyword}')")
                    s.faith = faith_id
                    print(f"[DEBUG] Session faith set to: '{s.faith}'. Exiting function.")
                    print("--- [DEBUG] Leaving try_set_faith ---\n")
                    return
    
    print("[DEBUG] Step 2 finished. No typo match found.")
    print("[DEBUG] No faith was set in this turn.")
    print("--- [DEBUG] Leaving try_set_faith ---\n")

def wants_retrieval(msg: str) -> bool:
    # This function remains the same
    m_lower = msg.lower()
    return any(word in m_lower for word in config.ASK_WORDS | config.DISTRESS_KEYWORDS)

def get_rag_context(msg: str, s: SessionState) -> Optional[str]:
    # This function remains the same
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
    # This function remains the same
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
    # This function remains the same
    if status == "refer-mental-health":
        text += "\n\n**Mandatory Referral:** If you are in crisis, please call or text 988 in the US/Canada, or find a local crisis line. Your safety is most important."
    elif status == "refer-faith":
        text += "\n\n**Note:** For deeper guidance, consider reaching out to a local faith leader or spiritual counselor."
    return text