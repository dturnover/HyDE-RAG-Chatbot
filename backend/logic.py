# logic.py
# Definitive version:
# 1. "Silver Lining" prompt for query rewrite.
# 2. Fixed RAG RULE for correct citation (no brackets).
# 3. Client vision (no "competitor" assumption, default faith, footer referral).
# 4. _edit_distance bug fix.
import re
from typing import Dict, List, Optional, Iterable
from dataclasses import dataclass, field
import config
import rag # Imports the updated rag module
import logging # Added logging

client = rag.client # Use the client initialized in rag.py

# --- Constants for Escalation (from config.py) ---
TURN_THRESHOLD_ESCALATE = 5 # Example: Escalate after 5 user messages
CRISIS_KEYWORDS = config.CRISIS_KEYWORDS

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
    if len(s1) > len(s2): s1, s2 = s2, s1
    distances = range(len(s1) + 1) # This line is now correct
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
            if abs(len(token) - len(keyword)) <= len_tolerance:
                if _edit_distance(token, keyword) <= max_diff:
                    return keyword
    return None

# --- Updated System Prompt (No Competitor Assumption, Refined Tone) ---
SYSTEM_BASE_FLOW = """You are the Fight Chaplain. Speak calmly and spiritually, like a trusted guide, using respectful, unisex language. Acknowledge the courage required for facing challenges. Your primary role is to listen empathetically and offer support grounded in faith when known. Start by acknowledging the user's current state and inviting them to share more."""

def system_message(s: SessionState, quote_allowed: bool, retrieval_ctx: Optional[str]) -> Dict[str, str]:
    """Generates the system message based on session state and RAG results."""
    rag_instruction = ""
    initial_response_guidance = "" # Keep responses concise initially

    # Refined Instruction based on conversation length (Step 1 adjustment)
    if s.turn_count <= 1 and not retrieval_ctx:
        initial_response_guidance = "Keep your initial responses very concise (1-2 sentences), focusing on listening and empathy."
    elif retrieval_ctx:
        initial_response_guidance = "The user has shared a concern and a relevant scripture was found. Respond with empathy and elaborate gently on how the retrieved verse might apply to their feeling."
    
    # Determine RAG instruction (Step 3)
    if quote_allowed and retrieval_ctx:
        # ★★★ THIS IS THE CITATION FIX ★★★
        # Removed the brackets [] from the example.
        rag_instruction = ("A relevant scripture is provided below. Weave a short, direct quote from the 'text' into your empathetic response, enclosed in quotes. "
                           "After the quote, you MUST cite the full 'Reference' field exactly as it is provided in the context, prefixed with an em dash (e.g., — The Retrieved Reference).")
    elif s.faith: 
        rag_instruction = (f"Their faith ({s.faith}) is known, but no scripture was retrieved. Respond with empathy and practical support, without mentioning scripture.")
    else:
        rag_instruction = ("Their faith is UNKNOWN. Do not provide scripture. Gently ask them to share their faith tradition if they are seeking scriptural support.")

    escalation_note = f"Escalation Status: {s.escalate_status}."
    session_status = f"Faith set={s.faith or 'bible_nrsv (Assumed)'}. User Turn={s.turn_count}. Quote allowed={quote_allowed and bool(retrieval_ctx)}. {escalation_note}"

    full_prompt = (f"{SYSTEM_BASE_FLOW} {initial_response_guidance}\n--- CONTEXT ---\n"
                   f"CURRENT SESSION STATUS: {session_status}\n"
                   f"RAG RULE: {rag_instruction}\n"
                   f"{retrieval_ctx or 'No passages retrieved.'}\n")
    return {"role": "system", "content": full_prompt}

# --- Faith Setting (Modified for Default Assumption) ---
def try_set_faith(msg: str, s: SessionState) -> bool:
    """Attempts to set faith based on keywords. Returns True if faith was newly set."""
    matched_keyword = _check_for_keywords_with_typo_tolerance(msg, config.FAITH_KEYWORDS.keys())
    if matched_keyword:
        new_faith = config.FAITH_KEYWORDS[matched_keyword]
        if s.faith != new_faith:
            s.faith = new_faith
            logging.info(f"[DEBUG logic.py] Faith explicitly set to: {s.faith} based on keyword '{matched_keyword}'")
            return True
    
    if s.faith is None:
        s.faith = "bible_nrsv" # Apply default Christian faith
        logging.info(f"[DEBUG logic.py] No faith specified, defaulting to: {s.faith}")
        return False
        
    return False

# --- Retrieval Trigger (Unchanged) ---
def wants_retrieval(msg: str) -> bool:
    """Checks if the message likely requests or implies a need for scripture."""
    all_trigger_keywords = config.ASK_WORDS | config.DISTRESS_KEYWORDS
    match = _check_for_keywords_with_typo_tolerance(msg, all_trigger_keywords)
    logging.info(f"[DEBUG logic.py] wants_retrieval check on '{msg[:50]}...': Match = {match}")
    return match is not None

# ★★★ RE-ENABLED "SILVER LINING" LLM REWRITE ★★★
def _get_rewritten_query(user_message: str) -> str:
    """Uses a fast LLM call to rewrite the user's query for better RAG results."""
    if not client:
         logging.info("[DEBUG logic.py] OpenAI client not available, returning original query.")
         return user_message
    
    logging.info(f"[DEBUG logic.py] Rewriting query: '{user_message}'")
    
    system_prompt = (
        "You are a search query transformation expert for a spiritual RAG chatbot. "
        "Your task is to convert a user's expression of distress into a query that finds a *solution-oriented* or *hopeful* passage. "
        "**Crucially, DO NOT** create a query that *mirrors* the negative emotion (e.g., 'verses about sadness'). Search for the *antidote*. "
        "**AVOID** themes of divine wrath or judgment. "
        "Examples:\n"
        "User: 'im depressed. i lost my fight yesterday'\n"
        "Rewrite: 'Scripture about finding hope after failure' or 'verses about healing and strength in sorrow'\n"
        "User: 'im nervous about my upcoming fight'\n"
        "Rewrite: 'verses about courage and finding strength in God' or 'scripture for peace and overcoming anxiety before a challenge'\n"
        "User: 'i'm so angry at my coach'\n"
        "Rewrite: 'scripture about patience and forgiveness' or 'verses on managing anger and finding peace'"
    )
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
            max_tokens=60,
            stream=False
        )
        rewritten = completion.choices[0].message.content
        rewritten_clean = rewritten.strip().strip('"').strip("'")
        logging.info(f"[DEBUG logic.py] Rewritten query: '{rewritten_clean}'")
        return rewritten_clean if rewritten_clean else user_message
    except Exception as e:
         logging.error(f"[DEBUG logic.py] ERROR during query rewrite: {e}")
         return user_message

# --- RAG Context Retrieval (Updated Faith Handling) ---
def get_rag_context(msg: str, s: SessionState) -> Optional[str]:
    """
    Determines if RAG is needed, gets the query, calls Pinecone search,
    and formats the result for the system prompt context. Uses default faith if needed.
    """
    if not s.faith:
        logging.error("[DEBUG logic.py] RAG check: Faith is None, THIS SHOULD NOT HAPPEN.")
        s.faith = "bible_nrsv" # Failsafe

    if wants_retrieval(msg):
        logging.info(f"[DEBUG logic.py] Retrieval triggered. Using faith: {s.faith}")
        search_query = _get_rewritten_query(msg)
        verse_text, verse_ref = rag.find_relevant_scripture(search_query, s.faith)

        if verse_text and verse_ref:
            logging.info(f"[DEBUG logic.py] RAG context generated: Ref='{verse_ref}', Text='{verse_text[:50]}...'")
            return f"RETRIEVED PASSAGE:\n- Reference: {verse_ref}\n- Text: \"{verse_text}\""
        else:
             logging.info("[DEBUG logic.py] rag.find_relevant_scripture returned no valid hit.")
             return None
    else:
        logging.info("[DEBUG logic.py] Retrieval not triggered.")
        return None

# --- Escalation & Metrics ---
def update_session_state(msg: str, s: SessionState) -> None:
    """
    Updates session state, including checking for escalation triggers.
    NOTE: This is called in main.py *after* try_set_faith.
    """
    logging.info(f"[DEBUG logic.py] Updating session state. Current turn: {s.turn_count}")
    if _check_for_keywords_with_typo_tolerance(msg, CRISIS_KEYWORDS):
        s.escalate_status = "crisis"
        logging.info(f"[DEBUG logic.py] CRISIS KEYWORD DETECTED. Status: 'crisis'.")
        return
    if s.turn_count >= TURN_THRESHOLD_ESCALATE and s.escalate_status == 'none':
        s.escalate_status = "needs_review"
        logging.info(f"[DEBUG logic.py] Turn threshold reached. Status: 'needs_review'.")

# --- Referral Footer (Step 6 & 7) ---
def apply_referral_footer(text: str, s: SessionState) -> str:
    """Appends appropriate referral based on escalation status AND standard offer."""
    footer = ""
    crisis_referral_added = False
    text = text.strip()

    if s.escalate_status == "crisis":
        footer += ("\n\nIt sounds like you're going through a very difficult time. For immediate support, "
                   "you can connect with people who can help by texting HOME to 741741 to reach the Crisis Text Line.")
        logging.info("[DEBUG logic.py] Appending crisis referral footer.")
        crisis_referral_added = True
    
    elif s.escalate_status == "needs_review":
        standard_offer = "Would you like help connecting with a faith leader from your tradition?"
        last_part = text[-len(standard_offer)*2:]
        if "faith leader" not in last_part.lower() and "spiritual leader" not in last_part.lower():
             footer += f"\n\n{standard_offer}"
             logging.info("[DEBUG logic.py] Appending standard faith leader referral (turn limit reached).")
        else:
             logging.info("[DEBUG logic.py] Skipping standard footer (turn limit), similar text already present.")

    return footer

