# logic.py
# FIX: Corrected UnboundLocalError in _edit_distance function.
# Contains all other recent changes (tone, default faith, referral logic).
import re
from typing import Dict, List, Optional, Iterable
from dataclasses import dataclass, field
import config
import rag # Imports the updated rag module

client = rag.client # Use the client initialized in rag.py

# --- Constants for Escalation (Ideally move to config.py) ---
TURN_THRESHOLD_ESCALATE = 10 # Example: Escalate after 10 turns (5 user messages)
CRISIS_KEYWORDS = {"suicide", "kill myself", "hopeless", "can't go on", "want to die"} # Example keywords

@dataclass
class SessionState:
    """A temporary object to hold state for a single request."""
    history: List[Dict[str, str]] = field(default_factory=list)
    faith: Optional[str] = None
    escalate_status: str = "none" # 'none', 'needs_review', 'crisis'

    @property
    def turn_count(self):
        user_turns = sum(1 for turn in self.history if turn.get("role") == "user")
        return user_turns # Count user messages

# --- Keyword/Typo Functions (BUG FIXED) ---
def _edit_distance(s1: str, s2: str) -> int:
    """Calculates the Levenshtein edit distance between two strings."""
    # ★★★ BUG FIX HERE ★★★
    # Original: if len(s1) > len(s2): s1, s2 = s2, s1; distances = range(len(s1) + 1)
    # This only initialized 'distances' if the 'if' was true.
    
    if len(s1) > len(s2): s1, s2 = s2, s1
    # FIX: Initialize 'distances' *after* the if statement, so it always runs.
    distances = range(len(s1) + 1)
    # ★★★ END BUG FIX ★★★

    for i2, c2 in enumerate(s2):
        new_distances = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2: new_distances.append(distances[i1])
            else: new_distances.append(1 + min((distances[i1], distances[i1 + 1], new_distances[-1])))
        distances = new_distances
    return distances[-1]

def _check_for_keywords_with_typo_tolerance(msg: str, keywords: Iterable[str]) -> Optional[str]:
    """Checks a message for a list of keywords with typo tolerance."""
    m = msg.lower()
    for keyword in keywords: # Exact match first
        if re.search(r'\b' + re.escape(keyword) + r'\b', m): return keyword
    msg_tokens = set(re.findall(r'\w+', m)) # Typo check second
    for keyword in keywords:
        max_diff = 1 if len(keyword) <= 6 else 2; len_tolerance = 2
        for token in msg_tokens:
            # Check for typos on words of similar length for efficiency
            if abs(len(token) - len(keyword)) <= len_tolerance:
                # Only call _edit_distance if length is plausible
                if _edit_distance(token, keyword) <= max_diff:
                    return keyword
    return None

# --- Updated System Prompt reflecting Step 1 Tone (No Competitor Assumption) ---
SYSTEM_BASE_FLOW = """You are the Fight Chaplain. Speak calmly and spiritually, like a trusted guide, using respectful, unisex language. Acknowledge the courage and grit required for facing challenges. Your primary role is to listen empathetically and offer support grounded in faith when known. Start by acknowledging the user's current state and inviting them to share more."""

def system_message(s: SessionState, quote_allowed: bool, retrieval_ctx: Optional[str]) -> Dict[str, str]:
    """Generates the system message based on session state and RAG results."""
    rag_instruction = ""
    initial_response_guidance = "" # Keep responses concise initially

    # Refined Instruction based on conversation length
    if s.turn_count <= 1: # Applies to first user message (turn_count 0) and second (turn_count 1)
        initial_response_guidance = "Keep your initial responses very concise (1-2 sentences), focusing on listening and empathy."

    # Determine RAG instruction (Step 3)
    if quote_allowed and retrieval_ctx:
        rag_instruction = ("A relevant scripture is provided below. Weave a short, direct quote from the 'text' into your empathetic response, enclosed in quotes. Cite it afterward (e.g., — Isaiah 41:10).")
    elif s.faith: # Faith is known (or assumed)
        rag_instruction = (f"Their faith ({s.faith}) is known, but no scripture was retrieved. Respond with empathy and practical support, without mentioning scripture.")
    else:
        # This block should now be rarely hit due to default assumption, but good fallback.
        rag_instruction = ("Their faith is UNKNOWN. Do not provide scripture. Gently ask them to share their faith tradition if they are seeking scriptural support.")


    escalation_note = f"Escalation Status: {s.escalate_status}."
    # Use effective faith (defaulting to bible_nrsv if None) for status display
    session_status = f"Faith set={s.faith or 'bible_nrsv (Assumed)'}. User Turn={s.turn_count}. Quote allowed={quote_allowed and bool(retrieval_ctx)}. {escalation_note}"

    # REMOVED REFERRAL RULE FROM HERE
    full_prompt = (f"{SYSTEM_BASE_FLOW} {initial_response_guidance}\n--- CONTEXT ---\n"
                   f"CURRENT SESSION STATUS: {session_status}\n"
                   f"RAG RULE: {rag_instruction}\n"
                   f"{retrieval_ctx or 'No passages retrieved.'}\n")
    return {"role": "system", "content": full_prompt}

# --- Faith Setting (Modified for Default Assumption) ---
def try_set_faith(msg: str, s: SessionState) -> bool:
    """Attempts to set faith based on keywords. Returns True if faith was newly set."""
    # Check for explicit faith keywords
    matched_keyword = _check_for_keywords_with_typo_tolerance(msg, config.FAITH_KEYWORDS.keys())
    if matched_keyword:
        new_faith = config.FAITH_KEYWORDS[matched_keyword]
        if s.faith != new_faith: # Check if it's actually changing
            s.faith = new_faith
            print(f"[DEBUG logic.py] Faith explicitly set to: {s.faith} based on keyword '{matched_keyword}'")
            return True
    
    # If no faith is set yet (it's None), apply default
    if s.faith is None:
        s.faith = "bible_nrsv" # Apply default Christian faith
        print(f"[DEBUG logic.py] No faith specified, defaulting to: {s.faith}")
        # Return False as it wasn't *explicitly* set by the user *this turn*
        return False 
        
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
    print(f"[DEBUG logic.py] Using original query for search: '{user_message}'")
    cleaned_message = user_message.strip().strip('"').strip("'")
    return cleaned_message

# --- RAG Context Retrieval (Updated Faith Handling) ---
def get_rag_context(msg: str, s: SessionState) -> Optional[str]:
    """
    Determines if RAG is needed, gets the query, calls Pinecone search,
    and formats the result for the system prompt context. Uses default faith if needed.
    """
    # Faith should be set (either explicitly or to default) by the time this runs
    # (Assuming main.py calls try_set_faith before this)
    if not s.faith:
        print("[DEBUG logic.py] RAG check: Faith is None, setting default.")
        s.faith = "bible_nrsv" # Failsafe default set

    if wants_retrieval(msg): # Check retrieval trigger first
        print(f"[DEBUG logic.py] Retrieval triggered. Using faith: {s.faith}")
        search_query = _get_rewritten_query(msg)
        verse_text, verse_ref = rag.find_relevant_scripture(search_query, s.faith) # Use the set faith

        if verse_text and verse_ref:
            print(f"[DEBUG logic.py] RAG context generated: Ref='{verse_ref}', Text='{verse_text[:50]}...'")
            return f"RETRIEVED PASSAGE:\n- Reference: {verse_ref}\n- Text: \"{verse_text}\""
        else:
             print("[DEBUG logic.py] rag.find_relevant_scripture returned no valid hit.")
             return None
    else:
        print("[DEBUG logic.py] Retrieval not triggered.")
        return None

# --- Escalation & Metrics (Unchanged Placeholder) ---
def update_session_state(msg: str, s: SessionState) -> None:
    """
    Updates session state, including checking for escalation triggers.
    NOTE: This should be called in main.py *after* try_set_faith.
    """
    print(f"[DEBUG logic.py] Updating session state. Current turn: {s.turn_count}")
    # Check for crisis keywords (Step 5 & 6 Trigger)
    if _check_for_keywords_with_typo_tolerance(msg, CRISIS_KEYWORDS):
        s.escalate_status = "crisis"
        print(f"[DEBUG logic.py] CRISIS KEYWORD DETECTED. Status: 'crisis'.")
        return
    # Check turn count threshold (Step 5 Trigger)
    if s.turn_count >= TURN_THRESHOLD_ESCALATE and s.escalate_status == 'none': # Only escalate once
        s.escalate_status = "needs_review"
        print(f"[DEBUG logic.py] Turn threshold reached. Status: 'needs_review'.")

# --- Referral Footer (Updated Logic - Step 6 & 7) ---
def apply_referral_footer(text: str, s: SessionState) -> str:
    """Appends appropriate referral based on escalation status AND standard offer."""
    footer = ""
    crisis_referral_added = False
    text = text.strip() # Ensure no trailing whitespace on LLM response

    # Specific referrals based on escalation (Step 6)
    if s.escalate_status == "crisis":
        footer += ("\n\nIt sounds like you're going through a very difficult time. For immediate support, "
                   "you can connect with people who can help by texting HOME to 741741 to reach the Crisis Text Line.")
        print("[DEBUG logic.py] Appending crisis referral footer.")
        crisis_referral_added = True
    
    # Standard Faith Leader Offer (Step 7) - Add ONLY if no crisis referral was given
    if not crisis_referral_added:
        standard_offer = "Would you like help connecting with a faith leader from your tradition?"
        # Check more robustly if similar offer already exists in the *last* part of the text
        last_part = text[-len(standard_offer)*2:] # Check last ~100 chars
        if "faith leader" not in last_part.lower() and "spiritual leader" not in last_part.lower():
             footer += f"\n\n{standard_offer}"
             print("[DEBUG logic.py] Appending standard faith leader referral footer.")
        else:
             print("[DEBUG logic.py] Skipping standard footer, similar text already present near end.")

    return text + footer

