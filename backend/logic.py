# logic.py
# Integrated client's 7-step vision into prompts and structure.
# Uses simple query passthrough (_get_rewritten_query).
import re
from typing import Dict, List, Optional, Iterable
from dataclasses import dataclass, field
import config
import rag # Imports the updated rag module

client = rag.client # Use the client initialized in rag.py

# --- Constants for Escalation (Ideally move to config.py) ---
TURN_THRESHOLD_ESCALATE = 10 # Example: Escalate after 10 turns (5 user messages)
CRISIS_KEYWORDS = {"suicide", "kill myself", "hopeless", "can't go on"} # Example keywords

@dataclass
class SessionState:
    """A temporary object to hold state for a single request."""
    history: List[Dict[str, str]] = field(default_factory=list)
    faith: Optional[str] = None
    escalate_status: str = "none" # 'none', 'needs_review', 'crisis'

    @property
    def turn_count(self):
        # A turn is considered a user message + assistant response pair
        return len(self.history) // 2

# --- Keyword/Typo Functions (Unchanged) ---
def _edit_distance(s1: str, s2: str) -> int:
    # ... (code unchanged) ...
    if len(s1) > len(s2): s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        new_distances = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2: new_distances.append(distances[i1])
            else: new_distances.append(1 + min((distances[i1], distances[i1 + 1], new_distances[-1])))
        distances = new_distances
    return distances[-1]

def _check_for_keywords_with_typo_tolerance(msg: str, keywords: Iterable[str]) -> Optional[str]:
    # ... (code unchanged) ...
    m = msg.lower()
    for keyword in keywords: # Exact match first
        if re.search(r'\b' + re.escape(keyword) + r'\b', m): return keyword
    msg_tokens = set(re.findall(r'\w+', m)) # Typo check second
    for keyword in keywords:
        max_diff = 1 if len(keyword) <= 6 else 2; len_tolerance = 2
        for token in msg_tokens:
            if abs(len(token) - len(keyword)) <= len_tolerance and _edit_distance(token, keyword) <= max_diff:
                return keyword
    return None

# --- Updated System Prompt reflecting Step 1 ---
SYSTEM_BASE_FLOW = """You are the Fight Chaplain. Address the user with respect, acknowledging their journey as a competitor requiring courage and grit. Speak calmly and spiritually, like a trusted guide, not a therapist or generic chatbot. Use only unisex language. Your role is to offer support grounded in faith and connect them to further resources when needed. Start by acknowledging their current state."""

def system_message(s: SessionState, quote_allowed: bool, retrieval_ctx: Optional[str]) -> Dict[str, str]:
    """Generates the system message based on session state and RAG results."""
    rag_instruction = ""
    referral_instruction = "" # Added instruction for step 7

    # Determine RAG instruction (Step 3)
    if quote_allowed and retrieval_ctx:
        rag_instruction = ("A relevant scripture has been provided below. Seamlessly weave a short, direct quote from the 'text' into your response, enclosed in quotes. Cite it afterward using an em dash (e.g., — Isaiah 41:10).")
    elif s.faith:
        rag_instruction = (f"Their faith ({s.faith}) is known, but no scripture was retrieved. Provide empathetic, practical support without mentioning scripture.")
    else:
        rag_instruction = ("Their faith is UNKNOWN. Do not provide scripture. If they seem to be seeking scriptural support, gently ask them to share their faith tradition.")

    # Determine Referral Instruction (Step 7) - Always offer faith leader connection
    referral_instruction = "Conclude your response by *always* offering to connect them with a faith leader from their tradition for deeper guidance."

    # Incorporate Escalation Status (Steps 4, 5, 6 influence context/tone implicitly)
    escalation_note = f"Escalation Status: {s.escalate_status}." # For potential LLM awareness

    session_status = f"Faith set={s.faith or 'None'}. Turn={s.turn_count}. Quote allowed={quote_allowed and bool(retrieval_ctx)}. {escalation_note}"
    full_prompt = (f"{SYSTEM_BASE_FLOW}\n--- CONTEXT ---\n"
                   f"CURRENT SESSION STATUS: {session_status}\n"
                   f"RAG RULE: {rag_instruction}\n"
                   f"REFERRAL RULE: {referral_instruction}\n" # Added referral rule
                   f"{retrieval_ctx or 'No passages retrieved.'}\n")
    return {"role": "system", "content": full_prompt}

# --- Faith Setting (Unchanged) ---
def try_set_faith(msg: str, s: SessionState) -> bool:
    """Attempts to set faith based on keywords. Returns True if faith was newly set."""
    if s.faith: return False
    matched_keyword = _check_for_keywords_with_typo_tolerance(msg, config.FAITH_KEYWORDS.keys())
    if matched_keyword:
        s.faith = config.FAITH_KEYWORDS[matched_keyword]
        print(f"[DEBUG logic.py] Faith set to: {s.faith} based on keyword '{matched_keyword}'")
        return True
    return False

# --- Retrieval Trigger (Unchanged) ---
def wants_retrieval(msg: str) -> bool:
    """Checks if the message likely requests or implies a need for scripture."""
    all_trigger_keywords = config.ASK_WORDS | config.DISTRESS_KEYWORDS
    match = _check_for_keywords_with_typo_tolerance(msg, all_trigger_keywords)
    print(f"[DEBUG logic.py] wants_retrieval check on '{msg[:50]}...': Match = {match}")
    return match is not None

# --- Query Passthrough (Unchanged) ---
def _get_rewritten_query(user_message: str) -> str:
    """
    Placeholder/Simplified version: Simply returns the original user message.
    """
    print(f"[DEBUG logic.py] Using original query for search (skipping LLM rewrite): '{user_message}'")
    cleaned_message = user_message.strip().strip('"').strip("'")
    return cleaned_message

# --- RAG Context Retrieval (Unchanged) ---
def get_rag_context(msg: str, s: SessionState) -> Optional[str]:
    """
    Determines if RAG is needed, gets the query, calls Pinecone search,
    and formats the result for the system prompt context.
    """
    if wants_retrieval(msg) and s.faith:
        print(f"[DEBUG logic.py] Retrieval triggered for faith: {s.faith}")
        search_query = _get_rewritten_query(msg)
        verse_text, verse_ref = rag.find_relevant_scripture(search_query, s.faith) # Assumes rag.py has debug prints

        if verse_text and verse_ref:
            print(f"[DEBUG logic.py] RAG context generated: Ref='{verse_ref}', Text='{verse_text[:50]}...'")
            return f"RETRIEVED PASSAGE:\n- Reference: {verse_ref}\n- Text: \"{verse_text}\""
        else:
             print("[DEBUG logic.py] rag.find_relevant_scripture returned no valid hit.")
             return None
    else:
        print("[DEBUG logic.py] Retrieval not triggered (faith unknown or no trigger words).")
        return None

# --- Escalation & Metrics (Placeholder Implementation - Step 4 & 5) ---
def update_session_state(msg: str, s: SessionState) -> None:
    """
    Updates session state, including checking for escalation triggers.
    NOTE: This might be better implemented in main.py *before* the LLM call.
    """
    print(f"[DEBUG logic.py] Updating session state. Current turn: {s.turn_count}")
    # Check for crisis keywords (Step 5 & 6 Trigger)
    if _check_for_keywords_with_typo_tolerance(msg, CRISIS_KEYWORDS):
        s.escalate_status = "crisis"
        print(f"[DEBUG logic.py] CRISIS KEYWORD DETECTED. Escalation set to 'crisis'.")
        return # Crisis overrides turn count

    # Check turn count threshold (Step 5 Trigger)
    if s.turn_count >= TURN_THRESHOLD_ESCALATE:
        s.escalate_status = "needs_review"
        print(f"[DEBUG logic.py] Turn threshold ({TURN_THRESHOLD_ESCALATE}) reached. Escalation set to 'needs_review'.")

# --- Referral Footer (Placeholder Implementation - Step 6) ---
def apply_referral_footer(text: str, s: SessionState) -> str:
    """Appends appropriate referral based on escalation status."""
    footer = ""
    # Always offer faith leader (Part of Step 7, handled in system prompt now, but could be reinforced here)

    # Specific referrals based on escalation (Step 6)
    if s.escalate_status == "crisis":
        # NOTE: Directly providing BetterHelp link might require specific partnership/legal review.
        # Using a generic mental health crisis text line is safer.
        footer += ("\n\nIt sounds like you're going through a very difficult time. For immediate support, "
                   "you can connect with people who can help by texting HOME to 741741 to reach the Crisis Text Line.")
        print("[DEBUG logic.py] Appending crisis referral footer.")
    elif s.escalate_status == "needs_review":
        # Generic offer for faith leader, handled by system prompt's REFERRAL RULE.
        # Could add specific text here if needed based on turn count.
        pass

    # Ensure the main LLM doesn't forget the standard faith leader offer (Step 7)
    # This might be redundant if the system prompt enforces it well.
    # standard_offer = "Would you like me to help connect you with a faith leader from your tradition?"
    # if standard_offer not in text and not footer: # Avoid adding if already present or crisis footer exists
    #      footer += f"\n\n{standard_offer}"

    return text + footer

# Conceptual flow in main.py would now involve update_session_state and apply_referral_footer
# Example:
# async def chat_endpoint(request: Request):
#     # ... [get data, setup SessionState s] ...
#     update_session_state(user_message, s) # Check escalation BEFORE getting LLM response
#     rag_ctx = get_rag_context(user_message, s)
#     system = system_message(s, bool(rag_ctx), rag_ctx)
#     messages = [system] + current_conversation
#     # ... [call LLM] ...
#     final_response_text = apply_referral_footer(llm_response_text, s) # Add footer AFTER getting LLM response
#     # ... [stream final_response_text] ...

