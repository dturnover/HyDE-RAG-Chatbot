# logic.py
#   This file is the brain of our application. While main.py is the server that
#   handles the requests. logic.py is where all the decisions are made

import re
from typing import Dict, List, Optional, Iterable
from dataclasses import dataclass, field
import config
import rag
import logging

client = rag.client

# --- constants ---
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



# --- session state ---

@dataclass
class SessionState:
    # a simple object to hold the user's current conversation state
    # we create a new one of these for every incoming message

    # the list of all past messages (e.g. [{"role": "user", "content": "hi"}])
    history: List[Dict[str, str]] = field(default_factory=list)

    # the user's detected faith
    faith: Optional[str] = None

    # the current safety status: "none", "needs_review", "crisis"
    escalate_status: str = "none"

    @property
    def turn_count(self):
        # calculates how many turns the user has taken
        user_turns = 0
        for turn in self.history:
            if turn.get("role") == "user":
                user_turns += 1
        return user_turns



# --- system prompt generation ---

# this is the base set of instructions for the AI
SYSTEM_BASE_FLOW = """You are the Fight Chaplain. You are a calm spiritual guide for combat sports athletes. You must speak like a guide, not a therapist or a chatbot, using warm, spiritually grounded, and concise unisex language (never "brother").

Your core mission is to set a tone of honor and open me (the user) to receive care.

To do this, you MUST follow this process:
1.  Acknowledge & Greet: Always begin *directly* by acknowledging my emotional or spiritual state. You must speak to my role as a warrior and the spiritual dimensions of competition, using fighter-oriented language (referencing courage, grit, and calling).
2.  Analyze: You must analyze my input to determine my emotional tone and spiritual readiness**. This is vital for **ensuring your relevance and credibility**.
3.  Be the Bridge: Always remember you are a bridge to a connection with a faith leader.
4.  Ask for Faith (Gently): If you don't know my faith and the moment feels right, you can gently ask if I (the user) am guided by a particular faith
"""

def system_message(s: SessionState, quote_allowed: bool, retrieval_ctx: Optional[str]) -> Dict[str, str]:
    # builds the final "system message" (AI's instructions) based on the current session state

    rag_instruction = ""

    # --- 1. build RAG instructions ---
    # if we found a verse, guide the AI
    if retrieval_ctx:
        # set the action instructions
        rag_instruction = (
            "Guidance: The user has shared a concern and a relevant scripture was found. Respond with empathy and elaborate gently on how the retrieved verse might apply to their feeling.\n\n"
            "Rule: A relevant scripture 'passage' is provided below. This 'passage' may contain both a text and its reference"
            "1. You MUST weave a short, direct quote from this 'passage' into your response"
            "2. CRITICAL: If the 'passage includes a citation (prefixed with -), you MUST include that citation exactly as provided at the end of your quote"
            "3. Do NOT separate the text from its citation. Do NOT invent a citation if one is not provided"
            "4. Never invent a quote or passage, even if the user asks for one. If no scripture is provided below, you MUST NOT provide one"
        )
    else:
        # set the action instruction
        rag_instruction = (
            f"Their faith ({FAITH_DISPLAY_NAMES.get(s.faith, s.faith)}) is known, but no scripture was retrieved"
            "Respond with empathy and practical support"
            "CRITICAL: Do NOT provide a scripture quote. Do NOT invent a quote. Do NOT make up a reference, even if the user asks"
            "politely support them without scripture"
        )

    # --- 2. build final prompt ---
    # combine all the pieces into one big prompt

    # a note for the AI on the current session status
    escalation_note = f"Escalation Status: {s.escalate_status}."
    session_status = (
        f"Faith set={s.faith or 'bible_nrsv (Assumed)'}. " 
        f"User Turn={s.turn_count}. "
        f"Quote allowed={quote_allowed and bool(retrieval_ctx)}. "
        f"{escalation_note}"
    )

    # combine all the parts
    full_prompt = (
        f"{SYSTEM_BASE_FLOW}\n"
        f"--- CONTEXT ---\n"
        f"CURRENT SESSION STATUS: {session_status}\n"
        f"RAG RULE: {rag_instruction}\n"
        f"{retrieval_ctx or 'No passages retrieved.'}\n"
    )

    # return the final system message in the format OpenAI expects
    return {"role": "system", "content": full_prompt}



# --- initialization function ---

def initialize_crisis_embeddings():
    # called once on server startup, pre-calculates embeddings for layer 2 semantic crisis so they're instantly ready for comparison

    logging.info("Initializing crisis phrase embeddings...")
    count = 0
    # loop through each phrase in our config list
    for phrase in config.CRISIS_PHRASES_SEMANTIC:
        # call the OpenAI API to get the embedding
        embedding = rag.get_embedding(phrase)
        if embedding:
            # store it in our global dictionary
            CRISIS_EMBEDDINGS[phrase] = embedding
            count += 1
    logging.info(f"successfully created {count} of {len(config.CRISIS_PHRASES_SEMANTIC)} crisis embeddings.")



# --- typo and keyword checking function ---

def edit_distance(s1, s2):
    # calculates the Levenshtein distance between two strings (number of edits required to change one word into another)

    # ensure s1 is the shorter string for efficiency
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    # distances is a list that will hold the edit distances
    distances = range(len(s1) + 1)

    # loop through each character in the longer string (s2)
    for i2, c2 in enumerate(s2):
        # start a new distances list for this row
        new_distances = [i2 + 1]

        # loop through each character in the shorter string (s1)
        for i1, c1 in enumerate(s1):
            # if the characters match, the cst is 0
            if c1 == c2:
                new_distances.append(distances[i1])
            # if they don't match the cost is 1
            else:
                new_distances.append(1 + min((distances[i1], distances[i1 + 1], new_distances[-1])))

        # update the 'distances' list for the next row
        distances = new_distances

    # the final value in the list is the edit distance
    return distances[-1]



def check_for_keywords(msg, keywords):
    # typo tolerant keyword checker that iterates through a list of keywords (iterable) and checks if any are present in the user's message

    # simple, fast, exact match search
    m = msg.lower()
    for keyword in keywords:
        # use regex to search for the keyword as a 'whole word'
        # this prevents 'us' from matching 'suicide'
        if re.search(r'\b' + re.escape(keyword) + r'\b', m):
            return keyword
        
    # slower, typo-tolerant match
    # get all individual words from the message
    msg_tokens = set(re.findall(r'\w+', m))

    for keyword in keywords:
        # we only check for typos on single words
        # if the keyword is a multi-word phrase skip it
        if " " in keyword:
            continue

        # set a dynamic typo allowance based on word length
        # short words (<= 6 chars) can have 1 typo
        # longer words (> 6 chars) can have 2 typos
        max_diff = 1 if len(keyword) <= 6 else 2

        # don't check words that are wildly different in length
        len_tolerance = 2

        # compare each word from the user's message
        for token in msg_tokens:
            # ... against the current keyword
            if abs(len(token) - len(keyword)) <= len_tolerance:
                # if the edit distance is within our allowance, its a match
                if edit_distance(token, keyword) <= max_diff:
                    return keyword # found a typo match, return it

    # if we get here no match was found
    return None



# --- state managemtn functions ---

def try_set_faith(msg, s):
    # checks the user's message for any faith keyword and updates the session state if a new faith is found

    # check the message against the FAITH_KEYWORDS dictionary keys
    matched_keyword = check_for_keywords(msg, config.FAITH_KEYWORDS.keys())

    if matched_keyword:
        # get the corresponding Pinecone index name (e.g. quran)
        new_faith = config.FAITH_KEYWORDS[matched_keyword]
        # only update if the faith has actually changed
        if s.faith != new_faith:
            s.faith = new_faith
            return True # return True to show we made an update

    # this is the default rule, if no faith has been set yet default to bible_nrsv
    if s.faith is None:
        s.faith = "bible_nrsv"
        return False # no new faith was found, it was just defaulted

    return False # no change was made



def wants_retrieval(msg):
    # checks if the user's message implies they want a scripture

    # combine both sets of keywords into one big set for checking
    all_trigger_keywords = config.ASK_WORDS | config.DISTRESS_KEYWORDS

    match = check_for_keywords(msg, all_trigger_keywords)

    # if we found any match return True
    return match is not None



def _get_hypothetical_document(user_message, faith):
    # implements HyDE: generates a hypothetical document for better RAG 
    # asks an AI to generate a hypothetical scripture that would be a better answer than the user's query
    # ew then use the hypothetical scripture to more effectively search Pinecone

    if not client:
        return user_message # fallbak if OpenAI client isn't ready

    # get the friendly name
    faith_display_name = FAITH_DISPLAY_NAMES.get(faith, "spiritual")

    # this is a speicial prompt just for the HyDE generation
    system_prompt = (
        f"You are a wise theologian. A user is feeling distressed. "
        f"Their expressed feeling is: '{user_message}'\n"
        f"Their faith tradition is: {faith_display_name}.\n\n"
        "Your task is to write a *single, hypothetical scripture passage* "
        "from their tradition that would provide the perfect comfort, hope, or "
        "strength for their situation. Write *only* the passage, as if it "
        "were a real quote. Do not add any commentary or labels like 'Hypothetical Verse:'. "
        "Focus on themes of hope, perseverance, and divine support. Pay special attention to the situation they describe themselves in"
    )

    try:
        # call the OpenAI API
        completion = client.chat.completions.create(
            model="gpt-4o-mini", # use a fast, cheap model forthis
            messages=[
                {"role": "system", "content": system_prompt},
            ],
            temperature=0.5,
            max_tokens=150,
            stream=False
        )
        hypothetical_doc = completion.choices[0].message.content

        # clean up any quotes or newlines the AI might have added
        hypothetical_doc_clean = hypothetical_doc.strip().strip('"').strip("'")
        logging.info(f"HyDE: Generated hypothetical doc: '{hypothetical_doc_clean[:100]}...'")
        
        # return the clean hypothetical doc or the original message if it failed
        return hypothetical_doc_clean if hypothetical_doc_clean else user_message

    except Exception as e:
        logging.error(f"ERROR during HyDE document generation: {e}")
        # fallback to the original message if the API call fails
        return user_message



def get_rag_context(msg, s):
    # this is the main RAG orchestration function
    # it decides how to search (HyDE vs raw) and then formats the result from
    # Pinecone into a single string to be "baked" into the system prompt
    # takes as input a msg str and s session state and returns a possibly empty str

    if not s.faith:
        # this should never happen because of our default rule in try_set_faith
        logging.error("RAG check: Faith is None, THIS SHOULD NOT HAPPEN.")
        s.faith = "bible_nrsv" # set default just in case

    # first, check if the user's message even warrants a search
    if wants_retrieval(msg):
        
        search_query = ""

        # --- smart RAG logic ---
        # check if the message contains emotional keywords
        is_distress = check_for_keywords(msg, config.DISTRESS_KEYWORDS)

        if is_distress:
            # it's an emotional request, use HyDE
            logging.info("RAG: Emotional request detected, using HyDE.")
            search_query = _get_hypothetical_document(msg, s.faith)
        else:
            # it's a topic request ("quote about...") use the raw message
            logging.info("RAG: Topic request detected, using raw message.")
            search_query = msg
        # --- end RAG logic ---

        # now call rag.py to find the actual scripture
        verse_text, verse_ref = rag.find_relevant_scripture(search_query, s.faith)

        # if we found a verse...
        if verse_text:

            # --- expanded citation cleaning ---
            if verse_ref:
                # clean the 'ref' string to make it look profressional before seen by LLM

                if s.faith=="gita":
                    # e.g. "Gita 1: Text 2" -> "Gita 1:2"
                    verse_ref = re.sub(r':\s*Text\s*', ':', verse_ref).replace("Gita ", "", 1)

                elif s.faith == "quran":
                    # e.g. "AK-INSHIRAH 94:6" -> "Qur'an, Al-Inshirah 94:6"
                    match = re.search(r'([A-Z\-]+) (\d+:\d+)', verse_ref, re.IGNORECASE)
                    if match:
                        chapter_name = match.group(1).title()
                        chapter_verse = match.group(2)
                        verse_ref = f"Qur'an, {chapter_name} {chapter_verse}"

                elif s.faith == "tanakh":
                    # e.g. "Tanakh: Genesis 1:1" -> "Genesis 1:1"
                    verse_ref = verse_ref.replace("Tanakh: ", "")

                elif s.faith == "dhammapada":
                    # e.g. "Dhammapada: Verse 1" -> "Dhammapada 1"
                    verse_ref = re.sub(r':\s*Verse\s*', ' ', verse_ref)

                # for bible_asv and nrsv the refs are usually clean so just trim whitespace
                else:
                    verse_ref = verse_ref.strip()
            # --- end cleaning ---

            # --- pre-baking the context string ---
            # combine the text and our cleaned reference into a perfectly formatted string for the AI
            full_passage = ""
            if verse_ref:
                full_passage = f"\"{verse_text}\" - {verse_ref}"
            else:
                # if no ref just send the text
                full_passage = f"\"{verse_text}\""

            # final string that gets added to the system prompt
            return f"RETRIEVED PASSAGE:\n- Passage: {full_passage}"

        else:
            # we wanted to search Pinecone but found nothing
            return None
    else:
        # the user's message didn't trigger RAG search
        return None



# --- layed escalation & referral functions ---

def update_session_state(msg, s):
    # updates escalation status using 2-layer safety check

    # --- Layer 1: Immediate keyword check (fast) ---
    # this checks our small, high-risk list from config.py
    if check_for_keywords(msg, config.CRISIS_KEYWORDS_IMMEDIATE):
        logging.warning(
            f"CRISIS DETECTED (Layer 1: Keyword Match): "
            f"Triggered by immediate-risk keyword."
        )
        s.escalate_status = "crisis"
        return # if layer 1 is hit stop immediately

    # --- Layer 2: Semantic Check (slower, AI-based) ---
    # this check compares the meaning of the user's message to our pre-calculated crisis phrase embeddings
    try:
        # get the embedding for th user's current message
        msg_embedding = rag.get_embedding(msg)

        if msg_embedding and CRISIS_EMBEDDINGS:
            # compare it against pre-loaded embeddings
            for phrase, crisis_embedding in CRISIS_EMBEDDINGS.items():

                # calculate the similarity (0.0 - 1.0)
                similarity = rag.get_cosine_similarity(msg_embedding, crisis_embedding)

                # if its too similar flag as crisis
                if similarity > CRISIS_SIMILARITY_THRESHOLD:
                    logging.warning(
                        f"CRISIS DETECTED (Layer 2: Semantic match):"
                        f"Similarity: {similarity:.2f} to cached phrase: '{phrase}"
                    )
                    s.escalate_status = "crisis"
                    return

    except Exception as e:
        # this can happen if OpenAI moderation blocks the embedding request
        logging.warning(F"CRISIS CHECK (Layer 2) FAILED: OpenAI moderation likely blocked the embedding. {e}")

    # --- Layer 3: turn-based escalation ---
    # if a conversation goes on too long offer to connect with faith leader
    if s.turn_count >= TURN_THRESHOLD_ESCALATE and s.escalate_status == 'none':
        s.escalate_status = "needs_review"
        logging.info(f"Turn threshold reached. Status: 'needs_review'.")


def apply_referral_footer(text: str, s):
    # checks the session status after the AI has generated a response and adds a crisis or referral footer if needed

    footer = ""
    text = text.strip()

    # 1. Crisis: always add the crisis text line
    if s.escalate_status == "crisis":
        footer += (
            "\n\nIt sounds like you're going through a very difficult time. For immediate support, "
            "you can connect with people who can help by texting HOME to 741741 to reach the Crisis Text Line."
        )

    # 2. Needs review: offer to connect with faith leader
    elif s.escalate_status == "needs_review":
        standard_offer = "Would you like help connecting with a faith leader from your tradition?"

        # check if the AI already said something similar to avoid repetion
        last_part = text[-len(standard_offer)*2:] # check the last 100 chars
        if "faith leader" not in last_part.lower() and "spiritual leader" not in last_part.lower():
            footer += f"\n\n{standard_offer}"

    return footer
