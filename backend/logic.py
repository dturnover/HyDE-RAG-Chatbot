# logic.py
#
# This is the final version. It *removes* the
# onboarding logic from the prompt, as the client's
# WordPress plugin already handles the first message.

import re
from typing import Dict, List, Optional, Iterable
from dataclasses import dataclass, field
import config
import rag
import logging

client = rag.client

# --- Constants ---

TURN_THRESHOLD_ESCALATE = 5
CRISIS_EMBEDDINGS: Dict[str, list[float]] = {}
CRISIS_SIMILARITY_THRESHOLD = 0.85

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
    history: List[Dict[str, str]] = field(default_factory=list)
    faith: Optional[str] = None
    escalate_status: str = "none"

    @property
    def turn_count(self):
        user_turns = sum(1 for turn in self.history if turn.get("role") == "user")
        return user_turns

# --- System Prompt Generation ---

# ★★★ THIS IS THE FINAL, SIMPLIFIED PROMPT ★★★
# It no longer has the "onboarding" or "warrior greeting" rules,
# as the client's plugin handles the first message.
SYSTEM_BASE_FLOW = """You are the Fight Chaplain. Speak *always* like a calm, spiritual guide for combat sports athletes. Your tone is respectful, grounded, **concise,** and uses unisex language.

**Your Core Process for Responding:**
1.  **Acknowledge the User:** Always begin by acknowledging their emotional or spiritual state. keep in mind they are a fighter/warrior and occassionally address them as such **SPARINGLY (once per 5 messages MAXIMUM)** don't overdo it or it sounds unnatural
2.  **Tone:** In *all* responses, weave in themes of **courage, grit, determination, spiritual calling** naturally, but *remain  somewhat concise*.
3.  **Guide, Not Bot:** Speak like a guide, NOT a therapist or a generic chatbot.
"""
# ★★★ END OF UPDATE ★★★

def system_message(s: SessionState, quote_allowed: bool, retrieval_ctx: Optional[str]) -> Dict[str, str]:
    """
    Builds the final "system message" (the AI's instructions)
    based on the current session state.
    """
    rag_instruction = ""
    initial_response_guidance = ""

    # We no longer need a special rule for turn_count <= 1
    if retrieval_ctx:
        initial_response_guidance = "The user has shared a concern and a relevant scripture was found. Respond with empathy and elaborate gently on how the retrieved verse might apply to their feeling."
    
    if quote_allowed and retrieval_ctx:
        rag_instruction = (
            "A relevant scripture 'Passage' is provided below. This 'Passage' may contain both a text and its reference."
            "1. You MUST weave a short, direct quote from this 'Passage' into your response. "
            "2. **CRITICAL:** If the 'Passage' includes a citation (prefixed with —), you MUST include that citation *exactly as provided* at the end of your quote. "
            "3. Do NOT separate the text from its citation. Do NOT invent a citation if one is not provided. "
            "4. NEVER invent a quote or passage, even if the user asks for one. If no scripture is provided below, you MUST NOT provide one."
        )
    else:
        # This block now correctly handles "no scripture found"
        # since s.faith will always have a default value.
        rag_instruction = (
            f"Their faith ({FAITH_DISPLAY_NAMES.get(s.faith, s.faith)}) is known, but no scripture was retrieved. "
            "Respond with empathy and practical support. "
            "**CRITICAL:** Do NOT provide a scripture quote. Do NOT invent a quote. Do NOT make up a reference, even if the user asks. "
            "Politely support them without scripture."
        )

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

# --- Initialization Function ---
def initialize_crisis_embeddings():
    """Called once on startup to pre-load semantic crisis phrases."""
    logging.info("Initializing crisis phrase embeddings...")
    count = 0
    for phrase in config.CRISIS_PHRASES_SEMANTIC:
        embedding = rag.get_embedding(phrase)
        if embedding:
            CRISIS_EMBEDDINGS[phrase] = embedding
            count += 1
    logging.info(f"Successfully created {count} of {len(config.CRISIS_PHRASES_SEMANTIC)} crisis embeddings.")

# --- Keyword & Typo Checking Functions ---

def _edit_distance(s1: str, s2: str) -> int:
    """Calculates Levenshtein distance for typo tolerance."""
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
    """Finds keywords in a message, allowing for typos."""
    m = msg.lower()
    for keyword in keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', m):
            return keyword
    msg_tokens = set(re.findall(r'\w+', m))
    for keyword in keywords:
        max_diff = 1 if len(keyword) <= 6 else 2
        len_tolerance = 2
        for token in msg_tokens:
            if abs(len(token) - len(keyword)) <= len_tolerance:
                if _edit_distance(token, keyword) <= max_diff:
                    return keyword
    return None

# --- State Management Functions ---

def try_set_faith(msg: str, s: SessionState) -> bool:
    """
    Checks message for faith keywords and updates session.
    If no faith is ever found, it defaults to "bible_nrsv".
    """
    matched_keyword = _check_for_keywords_with_typo_tolerance(msg, config.FAITH_KEYWORDS.keys())
    
    if matched_keyword:
        new_faith = config.FAITH_KEYWORDS[matched_keyword]
        if s.faith != new_faith:
            s.faith = new_faith
            return True
    
    # This is the default rule.
    if s.faith is None:
        s.faith = "bible_nrsv"
        return False
        
    return False

def wants_retrieval(msg: str) -> bool:
    """Checks if the message implies a desire for scripture."""
    all_trigger_keywords = config.ASK_WORDS | config.DISTRESS_KEYWORDS
    match = _check_for_keywords_with_typo_tolerance(msg, all_trigger_keywords)
    return match is not None

def _get_hypothetical_document(user_message: str, faith: str) -> str:
    """Implements HyDE: Generates a hypothetical doc for better RAG."""
    if not client:
        return user_message
    
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
    The main RAG function.
    Gets HyDE doc, searches Pinecone, cleans the citation,
    and "pre-bakes" it for the LLM.
    
    Uses HyDE for emotional requests and raw search for topic requests.
    """
    if not s.faith:
        logging.error("RAG check: Faith is None, THIS SHOULD NOT HAPPEN.")
        s.faith = "bible_nrsv"

    if wants_retrieval(msg):
        
        search_query = ""
        
        # --- "Smart RAG" Logic ---
        is_distress = _check_for_keywords_with_typo_tolerance(msg, config.DISTRESS_KEYWORDS)
        
        if is_distress:
            # It's an emotional request ("I'm sad..."). Use HyDE.
            logging.info("RAG: Emotional request detected, using HyDE.")
            search_query = _get_hypothetical_document(msg, s.faith)
        else:
            # It's just a topic request ("quote about..."). Use raw message.
            logging.info("RAG: Topic request detected, using raw message.")
            search_query = msg
        # --- End RAG Logic ---

        verse_text, verse_ref = rag.find_relevant_scripture(search_query, s.faith)

        if verse_text:
            # --- This is our Python citation cleaner ---
            if verse_ref:
                if s.faith == "gita":
                    # Cleans "Gita 1: Text 2" to "Gita 1:2"
                    verse_ref = re.sub(r':\s*Text\s*', ':', verse_ref).replace("Gita ", "", 1)
                
                elif s.faith == "quran":
                    # Cleans "AL-INSHIRAH 94:6" to "Qur'an, Al-Inshirah 94:6"
                    match = re.search(r'([A-Z\-]+) (\d+:\d+)', verse_ref, re.IGNORECASE)
                    if match:
                        chapter_name = match.group(1).title() # "AL-INSHIRAH" -> "Al-Inshirah"
                        chapter_verse = match.group(2)      # "94:6"
                        verse_ref = f"Qur'an, {chapter_name} {chapter_verse}"
                    # If it doesn't match, we leave it alone (it's prob already clean)

            # --- This is the "pre-baking" ---
            full_passage = ""
            if verse_ref:
                # Combine the text and our *cleaned* reference
                full_passage = f"\"{verse_text}\" — {verse_ref}"
            else:
                # If no ref, just send the text
                full_passage = f"\"{verse_text}\""
            
            # We now send a single, perfect "Passage" field to the LLM
            return f"RETRIEVED PASSAGE:\n- Passage: {full_passage}"
            # --- End "pre-baking" ---
        else:
            return None  # We wanted to search but found nothing
    else:
        return None  # The user's message didn't trigger a search

# --- Layered Escalation & Referral Functions ---

def update_session_state(msg: str, s: SessionState) -> None:
    """Updates escalation status using our 2-layer safety check."""
    
    # --- LAYER 1: Immediate Keyword Check (Fast, Typo-Tolerant) ---
    if _check_for_keywords_with_typo_tolerance(msg, config.CRISIS_KEYWORDS_IMMEDIATE):
        logging.warning(
            f"CRISIS DETECTED (Layer 1: Keyword Match): "
            f"Triggered by immediate-risk keyword."
        )
        s.escalate_status = "crisis"
        return

    # --- LAYER 2: Semantic Check (Slower, AI-based) ---
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
                    return
    except Exception as e:
        # This can happen if OpenAI moderation blocks the embedding request
        logging.warning(f"CRISIS CHECK (Layer 2) FAILED: OpenAI moderation likely blocked the embedding. {e}")
    
    # --- LAYER 3: Turn-Based Escalation ---
    if s.turn_count >= TURN_THRESHOLD_ESCALATE and s.escalate_status == 'none':
        s.escalate_status = "needs_review"
        logging.info(f"Turn threshold reached. Status: 'needs_review'.")

def apply_referral_footer(text: str, s: SessionState) -> str:
    """Adds crisis or faith leader footers if needed."""
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
        # Check if the AI *already* said something similar
        last_part = text[-len(standard_offer)*2:]
        if "faith leader" not in last_part.lower() and "spiritual leader" not in last_part.lower():
            footer += f"\n\n{standard_offer}"

    return footer