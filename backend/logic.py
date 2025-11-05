# logic.py
#
# This file holds all the "business logic" for the chatbot.
# It decides what to do with a user's message, manages the
# conversation's "state" (like escalation or faith), builds the
# instructions for the AI, and decides when to add referral footers.

import re  # For "regular expressions," used for advanced text matching
from typing import Dict, List, Optional, Iterable
from dataclasses import dataclass, field  # A handy tool for creating simple classes
import config  # We get all our keywords and settings from here
import rag  # For RAG functions and the OpenAI client
import logging  # For logging critical errors

# Use the OpenAI client that was already initialized in rag.py
client = rag.client

# --- Constants ---

# How many user turns before we suggest a human faith leader?
TURN_THRESHOLD_ESCALATE = 5

# A dictionary to hold our pre-calculated crisis embeddings
CRISIS_EMBEDDINGS: Dict[str, list[float]] = {}

# The sensitivity for our semantic crisis detection
CRISIS_SIMILARITY_THRESHOLD = 0.85

# Helper map to make faith names more human-readable for prompts
FAITH_DISPLAY_NAMES = {
    "bible_nrsv": "Christian (Bible)",
    "bible_asv": "Christian (Bible)",
    "tanakh": "Jewish (Tanakh)",
    "quran": "Muslim (Quran)",
    "gita": "Hindu (Bhagavad Gita)",
    "dhammapada": "Buddhist (Dhammapada)",
}

# --- Session State ---

@dataclass
class SessionState:
    """
    A simple "container" to hold information about a single chat session.
    We create a new one of these for every single request.
    """
    history: List[Dict[str, str]] = field(default_factory=list)
    faith: Optional[str] = None  # e.g., "bible_nrsv", "quran", "gita"
    escalate_status: str = "none"  # Can be "none", "needs_review", or "crisis"

    @property
    def turn_count(self):
        """Calculates how many turns the *user* has taken."""
        user_turns = sum(1 for turn in self.history if turn.get("role") == "user")
        return user_turns

# --- Initialization Function ---

def initialize_crisis_embeddings():
    """
    Called once by main.py on server startup.
    This calculates the embeddings for all our semantic crisis phrases
    and stores them in memory for fast comparison.
    """
    logging.info("Initializing crisis phrase embeddings...")
    count = 0
    # Use the semantic phrase list from config
    for phrase in config.CRISIS_PHRASES_SEMANTIC:
        embedding = rag.get_embedding(phrase)
        if embedding:
            CRISIS_EMBEDDINGS[phrase] = embedding
            count += 1
    logging.info(f"Successfully created {count} of {len(config.CRISIS_PHRASES_SEMANTIC)} crisis embeddings.")

# --- Keyword & Typo Checking Functions ---

def _edit_distance(s1: str, s2: str) -> int:
    """
    Calculates the "Levenshtein distance" between two strings.
    This is a fancy way of saying "how many single-character
    edits (inserts, deletes, substitutions) would it take
    to change s1 into s2?"
    
    We use this to catch typos (e.g., "depresed" vs "depressed").
    """
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

def _check_for_keywords_with_typo_tolerance(msg: str, keywords: Iterable[str]) -> Optional[str]:
    """
    Checks a message for a list of keywords, allowing for small typos.
    
    It first checks for an exact match. If it doesn't find one,
    it breaks the message into individual words and checks if any
    word is "close enough" (using _edit_distance) to a keyword.
    """
    m = msg.lower()
    
    # 1. Exact match (fast path)
    for keyword in keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', m):
            return keyword
            
    # 2. Typo check (slower path)
    msg_tokens = set(re.findall(r'\w+', m))  # Get all unique words
    for keyword in keywords:
        max_diff = 1 if len(keyword) <= 6 else 2
        len_tolerance = 2
        
        for token in msg_tokens:
            if abs(len(token) - len(keyword)) <= len_tolerance:
                if _edit_distance(token, keyword) <= max_diff:
                    return keyword  # Found a close-enough match
    return None

# --- System Prompt Generation ---

# This is the base instruction for the AI, setting its personality.
SYSTEM_BASE_FLOW = """You are the Fight Chaplain. Speak calmly and spiritually, like a trusted guide, using respectful, unisex language. Acknowledge the courage required for facing challenges. Your primary role is to listen empathetically and offer support grounded in faith when known. Start by acknowledging the user's current state and inviting them to share more."""

def system_message(s: SessionState, quote_allowed: bool, retrieval_ctx: Optional[str]) -> Dict[str, str]:
    """
    Builds the final "system message" (the AI's instructions)
    based on the current session state.
    """
    rag_instruction = ""
    initial_response_guidance = ""

    if s.turn_count <= 1 and not retrieval_ctx:
        initial_response_guidance = "Keep your initial responses very concise (1-2 sentences), focusing on listening and empathy."
    elif retrieval_ctx:
        initial_response_guidance = "The user has shared a concern and a relevant scripture was found. Respond with empathy and elaborate gently on how the retrieved verse might apply to their feeling."
    
    # --- ★★★ UPDATED: Stricter RAG/Hallucination Fix ★★★ ---
    if quote_allowed and retrieval_ctx:
        # We found a quote and are allowed to use it.
        # This rule is now extremely strict to fix the citation and hallucination failures.
        rag_instruction = (
            "A relevant scripture passage (Text and Reference) is provided below. "
            "1. You MUST weave a short, direct quote from the 'Text' field into your response, enclosed in quotes. "
            "2. If a 'Reference' field is provided and is not empty, you MUST cite it *exactly* as it appears, prefixed with an em dash (—). "
            "3. If the 'Reference' field is missing or empty, do NOT cite anything. Do NOT make up a reference. "
            "4. NEVER invent a quote or passage, even if the user asks for one. If no scripture is provided below, you MUST NOT provide one."
        )
    elif s.faith:
        # We know their faith, but didn't find a quote
        rag_instruction = (
            f"Their faith ({FAITH_DISPLAY_NAMES.get(s.faith, s.faith)}) is known, but no scripture was retrieved. "
            "Respond with empathy and practical support. "
            "**CRITICAL:** Do NOT provide a scripture quote. Do NOT invent a quote. Do NOT make up a reference, even if the user asks. "
            "Politely support them without scripture."
        )
    else:
        # We don't know their faith
        rag_instruction = (
            "Their faith is UNKNOWN. Do not provide scripture. "
            "Gently ask them to share their faith tradition if they are seeking scriptural support. "
            "Do NOT invent a quote."
        )
    # --- End of Fix ---

    escalation_note = f"Escalation Status: {s.escalate_status}."
    session_status = (
        f"Faith set={s.faith or 'bible_nrsv (Assumed)'}. "
        f"User Turn={s.turn_count}. "
        f"Quote allowed={quote_allowed and bool(retrieval_ctx)}. "
        f"{escalation_note}"
    )

    full_prompt = (
        f"{SYSTEM_BASE_FLOW} {initial_response_guidance}\n"
        f"--- CONTEXT ---\n"
        f"CURRENT SESSION STATUS: {session_status}\n"
        f"RAG RULE: {rag_instruction}\n"
        f"{retrieval_ctx or 'No passages retrieved.'}\n"
    )
    
    return {"role": "system", "content": full_prompt}

# --- State Management Functions ---

def try_set_faith(msg: str, s: SessionState) -> bool:
    """
    Checks the user's message for any faith-related keywords.
    """
    matched_keyword = _check_for_keywords_with_typo_tolerance(msg, config.FAITH_KEYWORDS.keys())
    
    if matched_keyword:
        new_faith = config.FAITH_KEYWORDS[matched_keyword]
        if s.faith != new_faith:
            s.faith = new_faith
            return True
    
    if s.faith is None:
        s.faith = "bible_nrsv"  # Default to Christian (NRSV)
        return False
        
    return False

def wants_retrieval(msg: str) -> bool:
    """
    Checks if the user's message implies they want scripture.
    This is a (fast) keyword check.
    """
    all_trigger_keywords = config.ASK_WORDS | config.DISTRESS_KEYWORDS
    match = _check_for_keywords_with_typo_tolerance(msg, all_trigger_keywords)
    return match is not None

def _get_hypothetical_document(user_message: str, faith: str) -> str:
    """
    This implements the HyDE (Hypothetical Document Embeddings) technique.
    We ask the AI to write a *hypothetical scripture verse* that would
    perfectly comfort the user.
    """
    if not client:
        return user_message  # Fallback if OpenAI client fails
    
    faith_display_name = FAITH_DISPLAY_NAMES.get(faith, "spiritual")

    system_prompt = (
        f"You are a wise theologian. A user is feeling distressed. "
        f"Their expressed feeling is: '{user_message}'\n"
        f"Their faith tradition is: {faith_display_name}.\n\n"
        "Your task is to write a *single, hypothetical scripture passage* "
        "from their tradition that would provide the perfect comfort, hope, or "
        "strength for their situation. Write *only* the passage, as if it "
        "were a real quote. Do not add any commentary or labels like 'Hypothetical Verse:'. "
        "Focus on themes of hope, perseverance, and divine support."
    )
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
            ],
            temperature=0.0,
            max_tokens=150,
            stream=False
        )
        hypothetical_doc = completion.choices[0].message.content
        hypothetical_doc_clean = hypothetical_doc.strip().strip('"').strip("'")
        
        logging.info(f"HyDE: Generated hypothetical doc: '{hypothetical_doc_clean[:100]}...'")
        
        return hypothetical_doc_clean if hypothetical_doc_clean else user_message
    except Exception as e:
        logging.error(f"ERROR during HyDE document generation: {e}")
        return user_message

def get_rag_context(msg: str, s: SessionState) -> Optional[str]:
    """
    The main function to decide if and what to retrieve from Pinecone.
    """
    if not s.faith:
        logging.error("RAG check: Faith is None, THIS SHOULD NOT HAPPEN.")
        s.faith = "bible_nrsv"

    if wants_retrieval(msg):
        search_document = _get_hypothetical_document(msg, s.faith)
        verse_text, verse_ref = rag.find_relevant_scripture(search_document, s.faith)

        # ★★★ UPDATED: Dhammapada/Citation Fix ★★★
        # We now check if the reference is valid. If not, we still
        # send the text, but the new system_message rule will
        # know to simply *not* cite it.
        if verse_text:
            # Clean up the reference if it exists
            if verse_ref and s.faith == "gita":
                verse_ref = re.sub(r':\s*Text\s*', ':', verse_ref).replace("Gita ", "", 1)
            
            # Format the context for the prompt
            # The new prompt logic handles the `verse_ref` being None
            return f"RETRIEVED PASSAGE:\n- Reference: {verse_ref}\n- Text: \"{verse_text}\""
        else:
            return None  # We wanted to search but found nothing
    else:
        return None  # The user's message didn't trigger a search

# --- ★★★ UPDATED: Layered Escalation & Referral Functions ★★★ ---

def update_session_state(msg: str, s: SessionState) -> None:
    """
    Updates the session's escalation status using a new
    two-layer check to prevent OpenAI moderation from blocking.
    """
    
    # --- LAYER 1: Immediate Keyword Check (Fast, Typo-Tolerant, Unblockable) ---
    # We use our original typo-check function on the new immediate-risk list.
    if _check_for_keywords_with_typo_tolerance(msg, config.CRISIS_KEYWORDS_IMMEDIATE):
        logging.warning(
            f"CRISIS DETECTED (Layer 1: Keyword Match): "
            f"Triggered by immediate-risk keyword."
        )
        s.escalate_status = "crisis"
        return  # Crisis status overrides all other states

    # --- LAYER 2: Semantic Check (Slower, Subtle, AI-based) ---
    # This only runs if Layer 1 passes.
    # This *can* be blocked by OpenAI, but Layer 1 is our safety net.
    try:
        msg_embedding = rag.get_embedding(msg)
        
        if msg_embedding and CRISIS_EMBEDDINGS:
            for phrase, crisis_embedding in CRISIS_EMBEDDINGS.items():
                
                similarity = rag.get_cosine_similarity(msg_embedding, crisis_embedding)
                
                if similarity > CRISIS_SIMILARITY_THRESHOLD:
                    logging.warning(
                        f"CRISIS DETECTED (Layer 2: Semantic Match): "
                        f"Similarity: {similarity:.2f} to cached phrase: '{phrase}'"
                    )
                    s.escalate_status = "crisis"
                    return  # Crisis status overrides all other states
    except Exception as e:
        # This can happen if OpenAI's moderation *blocks* the embedding request
        logging.warning(f"CRISIS CHECK (Layer 2) FAILED: OpenAI moderation likely blocked the embedding. {e}")
        # We don't escalate here, we just fall through to the turn-based check.
        # Layer 1 has already caught the most dangerous phrases.
    
    # --- LAYER 3: Turn-Based Escalation ---
    if s.turn_count >= TURN_THRESHOLD_ESCALATE and s.escalate_status == 'none':
        s.escalate_status = "needs_review"
        logging.info(f"Turn threshold reached. Status: 'needs_review'.")

def apply_referral_footer(text: str, s: SessionState) -> str:
    """
    Checks the final AI response and session state to see if we
    need to add a footer message.
    """
    footer = ""
    text = text.strip()

    # 1. Crisis: Always add the crisis text line
    if s.escalate_status == "crisis":
        footer += (
            "\n\nIt sounds like you're going through a very difficult time. For immediate support, "
            "you can connect with people who can help by texting HOME to 741741 to reach the Crisis Text Line."
        )
        
    # 2. Needs Review: Offer to connect to a human
    elif s.escalate_status == "needs_review":
        standard_offer = "Would you like help connecting with a faith leader from your tradition?"
        
        last_part = text[-len(standard_offer)*2:]
        if "faith leader" not in last_part.lower() and "spiritual leader" not in last_part.lower():
            footer += f"\n\n{standard_offer}"

    return footer