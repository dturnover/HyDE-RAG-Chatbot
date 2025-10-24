# tools/chunker.py
# Final Correction: Fixed Dhammapada parser to use ASCII hyphen in IDs.
import sys
import json
import re
import argparse
import os
from typing import Dict, TextIO # Corrected import

# --- Data for Reference Checks ---
# ... (BIBLE_CHAPTER_LIMITS, QURAN_CHAPTER_LIMITS - unchanged) ...
TANAKH_BOOKS = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Joshua", "Judges", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings",
    "Isaiah", "Jeremiah", "Ezekiel", "Hosea", "Joel", "Amos", "Obadiah",
    "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai",
    "Zechariah", "Malachi", "Psalms", "Proverbs", "Job", "Song of Songs",
    "Ruth", "Lamentations", "Ecclesiastes", "Esther", "Daniel",
    "Ezra", "Nehemiah", "1 Chronicles", "2 Chronicles"
]
TANAKH_BOOKS_LOWER = {b.lower() for b in TANAKH_BOOKS}
BIBLE_CHAPTER_LIMITS = {
    # Old Testament (using common Protestant canon order for simplicity)
    "Genesis": 50, "Exodus": 40, "Leviticus": 27, "Numbers": 36, "Deuteronomy": 34,
    "Joshua": 24, "Judges": 21, "Ruth": 4, "1 Samuel": 31, "2 Samuel": 24,
    "1 Kings": 22, "2 Kings": 25, "1 Chronicles": 29, "2 Chronicles": 36,
    "Ezra": 10, "Nehemiah": 13, "Esther": 10, "Job": 42, "Psalms": 150,
    "Proverbs": 31, "Ecclesiastes": 12, "Song of Songs": 8, "Song of Solomon": 8, # Alias
    "Isaiah": 66, "Jeremiah": 52, "Lamentations": 5, "Ezekiel": 48, "Daniel": 12,
    "Hosea": 14, "Joel": 3, "Amos": 9, "Obadiah": 1, "Jonah": 4, "Micah": 7,
    "Nahum": 3, "Habakkuk": 3, "Zephaniah": 3, "Haggai": 2, "Zechariah": 14, "Malachi": 4,
    # New Testament
    "Matthew": 28, "Mark": 16, "Luke": 24, "John": 21, "Acts": 28,
    "Romans": 16, "1 Corinthians": 16, "2 Corinthians": 13, "Galatians": 6,
    "Ephesians": 6, "Philippians": 4, "Colossians": 4, "1 Thessalonians": 5,
    "2 Thessalonians": 3, "1 Timothy": 6, "2 Timothy": 4, "Titus": 3, "Philemon": 1,
    "Hebrews": 13, "James": 5, "1 Peter": 5, "2 Peter": 3, "1 John": 5, "2 John": 1,
    "3 John": 1, "Jude": 1, "Revelation": 22,
    # Apocrypha/Deuterocanonical (Add more as needed based on NRSV content)
    "Tobit": 14, "Judith": 16, "Add Esther": 1, "Wis": 19, "Wisdom": 19, "Wisdom of Solomon": 19,# Alias
    "Sir": 51, "Sirach": 51, "Ecclesiasticus": 51,# Alias
    "Bar": 6, "Baruch": 6, # Alias
    "1 Esd": 9, "1 Esdras": 9, # Alias
    "2 Esd": 16, "2 Esdras": 16, # Alias
    "Let Jer": 1, "Letter of Jeremiah": 1, # Alias
    "Song of Thr": 1, "Song of Three": 1, # Alias
    "Sus": 1, "Susanna": 1, # Alias
    "Bel": 1, "Bel and the Dragon": 1,# Alias
    "1 Macc": 16, "1 Maccabees": 16, # Alias
    "2 Macc": 15, "2 Maccabees": 15, # Alias
    "3 Macc": 7, "3 Maccabees": 7, # Alias
    "4 Macc": 18, "4 Maccabees": 18, # Alias
    "Pr Man": 1, "Prayer of Manasseh": 1, # Alias
    # Placeholder for Gita/Dhammapada if reference format were standard C:V
    "Gita": 18, # Gita Chapters (approx)
    "Dhammapada": 26 # Dhammapada Chapters (approx)
}
QURAN_CHAPTER_LIMITS = 114 # Max Surah number


# --- Utility Functions ---
def write(out: TextIO, row: Dict): # Added TextIO hint back
    """Writes a dictionary as a JSON line to the output file."""
    out.write(json.dumps(row, ensure_ascii=False) + "\n")

# ---------- 6 DEDICATED PARSERS ----------

# --- 1. Parser for corpus/ASVHB.txt ---
# (Confirmed working - No changes needed)
def parse_asv_bible(inp: str, source: str, out: TextIO):
    print(f"Using dedicated parser: parse_asv_bible...")
    # ... [rest of parse_asv_bible - unchanged] ...
    cur_book = ""
    current_ref_parts = None
    current_text = ""
    book_header_re = re.compile(r'^\s*===\s*([^=]+)\s*===\s*$')
    cv_marker_re = re.compile(r'\{(\d+):(\d+)\}')
    count = 0
    with open(inp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            book_match = book_header_re.match(line)
            if book_match:
                if current_ref_parts:
                    book, chap, verse = current_ref_parts
                    cleaned_text = re.sub(r'\s+', ' ', current_text).strip()
                    if cleaned_text:
                        row = { "id": f"{source}-{book}-{chap}-{verse}", "source": source, "ref": f"{book} {chap}:{verse}", "book": book, "chapter": chap, "verse": verse, "text": cleaned_text }
                        write(out, row)
                        count += 1
                cur_book = book_match.group(1).strip().title()
                current_ref_parts = None
                current_text = ""
                continue
            if not cur_book: continue

            matches = list(cv_marker_re.finditer(line))
            if not matches:
                if current_ref_parts: current_text += " " + line.strip()
                continue

            line_cursor = 0
            for match in matches:
                text_slice = line[line_cursor:match.start()].strip()
                if current_text is not None: current_text += " " + text_slice
                else: current_text = text_slice

                if current_ref_parts:
                    book, chap, verse = current_ref_parts
                    cleaned_text = re.sub(r'\s+', ' ', current_text).strip()
                    if cleaned_text:
                        row = { "id": f"{source}-{book}-{chap}-{verse}", "source": source, "ref": f"{book} {chap}:{verse}", "book": book, "chapter": chap, "verse": verse, "text": cleaned_text }
                        write(out, row)
                        count += 1

                chap, verse = int(match.group(1)), int(match.group(2))
                current_ref_parts = (cur_book, chap, verse)
                current_text = ""
                line_cursor = match.end()

            remaining_text = line[line_cursor:].strip()
            if current_text is not None: current_text += " " + remaining_text
            else: current_text = remaining_text

    if current_ref_parts:
        book, chap, verse = current_ref_parts
        cleaned_text = re.sub(r'\s+', ' ', current_text).strip()
        if cleaned_text:
            row = { "id": f"{source}-{book}-{chap}-{verse}", "source": source, "ref": f"{book} {chap}:{verse}", "book": book, "chapter": chap, "verse": verse, "text": cleaned_text }
            write(out, row)
            count += 1
    print(f"  > Wrote {count} records.")


# --- 2. Parser for corpus/nrsv_bible.txt ---
# (Confirmed working - No changes needed)
def parse_nrsv_bible(inp: str, source: str, out: TextIO):
    print(f"Using dedicated parser: parse_nrsv_bible (FINAL)...")
    cur_book = ""
    cur_chapter = 0
    current_verse_num = 0 # Track the *last* verse number found
    # Stricter regex for chapter markers: [BookName ChapterNumOrRoman]
    chapter_marker_re = re.compile(r'^\s*\[([A-Za-z\s.-]+?)\s+([0-9ivxlcdm]+)\]\s*$', re.IGNORECASE)
    # Regex for verse lines starting with a number (captures number and text separately)
    verse_line_re = re.compile(r'^\s*(\d+)\s+(.*)')
    # Regex to find verse numbers starting a line OR preceded by space
    verse_num_finder_re = re.compile(r'(?:^|\s)(\d+)\s')
    count = 0
    found_first_chapter = False
    current_line_text_buffer = "" # Accumulate text across lines until a new verse/chapter

    try:
        with open(inp, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f):
                line_stripped = line.strip()
                if not line_stripped: continue

                chap_match = chapter_marker_re.match(line_stripped)
                if chap_match:
                    if found_first_chapter and cur_book and cur_chapter > 0 and current_verse_num > 0 and current_line_text_buffer:
                        cleaned_text = re.sub(r'\s+', ' ', current_line_text_buffer).strip()
                        if current_verse_num > 0:
                            row = { "id": f"{source}-{cur_book}-{cur_chapter}-{current_verse_num}", "source": source, "ref": f"{cur_book} {cur_chapter}:{current_verse_num}", "book": cur_book, "chapter": cur_chapter, "verse": current_verse_num, "text": cleaned_text }
                            write(out, row); count += 1
                    found_first_chapter = True
                    book_name, chap_str = chap_match.groups()
                    cur_book = book_name.strip().title()
                    cur_book = re.sub(r'^(\d)\s', r'\1 ', cur_book)
                    if chap_str.lower() == 'i': cur_chapter = 1
                    else:
                         try: cur_chapter = int(chap_str)
                         except ValueError:
                              roman_map={'i':1,'v':5,'x':10,'l':50,'c':100,'d':500,'m':1000}; val=0; prev_val=0; valid=True
                              for n in reversed(chap_str.lower()):
                                   if n not in roman_map: valid=False; break
                                   curr=roman_map[n]; val += curr if curr>=prev_val else -curr; prev_val=curr
                              if valid: cur_chapter = val
                              else: print(f"> WARN: Bad chap '{chap_str}' L{line_num+1}"); cur_chapter=0
                    current_verse_num = 0; current_line_text_buffer = ""
                    continue

                if not found_first_chapter: continue # Skip lines until first chapter found

                line_cursor = 0
                found_verse_on_line = False
                for match in verse_num_finder_re.finditer(line_stripped):
                    if match.start() == 0 or line_stripped[match.start()-1].isspace():
                        found_verse_on_line = True
                        verse_num_str = match.group(1) # Group 1 is the number

                        text_slice = line_stripped[line_cursor:match.start()].strip()
                        if current_verse_num > 0: current_line_text_buffer += " " + text_slice
                        elif text_slice: print(f"  > WARNING: Discarding text before first verse '{verse_num_str}': '{text_slice[:50]}...' L{line_num+1}")

                        if current_verse_num > 0 and current_line_text_buffer:
                            cleaned_text = re.sub(r'\s+', ' ', current_line_text_buffer).strip()
                            row = { "id": f"{source}-{cur_book}-{cur_chapter}-{current_verse_num}", "source": source, "ref": f"{cur_book} {cur_chapter}:{current_verse_num}", "book": cur_book, "chapter": cur_chapter, "verse": current_verse_num, "text": cleaned_text }
                            write(out, row); count += 1

                        try:
                            current_verse_num = int(verse_num_str)
                            current_line_text_buffer = "" # Reset buffer
                            line_cursor = match.end() # Move cursor past number+space
                        except ValueError:
                            print(f"  > WARNING: Invalid verse num '{verse_num_str}' L{line_num+1}. Appending rest.")
                            current_line_text_buffer += " " + line_stripped[line_cursor:]
                            line_cursor = len(line_stripped)
                            current_verse_num = 0; break # Stop processing line

                remaining_text = line_stripped[line_cursor:].strip()
                if remaining_text: current_line_text_buffer += " " + remaining_text
                if not found_verse_on_line and current_verse_num > 0: # Pure continuation line
                     current_line_text_buffer += " " + line_stripped

    except Exception as e: print(f"  > FATAL ERROR during file processing: {e}")

    if cur_book and cur_chapter > 0 and current_verse_num > 0 and current_line_text_buffer:
        cleaned_text = re.sub(r'\s+', ' ', current_line_text_buffer).strip()
        row = { "id": f"{source}-{cur_book}-{cur_chapter}-{current_verse_num}", "source": source, "ref": f"{cur_book} {cur_chapter}:{current_verse_num}", "book": cur_book, "chapter": cur_chapter, "verse": current_verse_num, "text": cleaned_text }
        write(out, row); count += 1

    if count == 0: print(f"  > FATAL ERROR: Wrote 0 records.")
    else: print(f"  > Wrote {count} records.")


# --- 3. Parser for corpus/tanakh_et.txt ---
# (Confirmed working with warnings - Needs final check after fixes elsewhere)
def parse_tanakh(inp: str, source: str, out: TextIO):
    print(f"Using dedicated parser: parse_tanakh (Reference Fix v2)...")
    cur_book = ""
    cv_line_pattern = re.compile(r'^\s*(\d+),(\d+)\s*$')
    in_scripture = False
    count = 0
    pending_cv = None
    last_line_was_cv = False

    with open(inp, "r", encoding="utf-8", errors="ignore") as f:
        # ... [rest of parse_tanakh - unchanged] ...
        for line_num, line in enumerate(f):
            line_stripped = line.strip()
            if not line_stripped:
                last_line_was_cv = False
                continue

            if not in_scripture:
                if line_stripped.lower() == "genesis":
                     in_scripture = True; cur_book = "Genesis"; last_line_was_cv = False
                     continue
                else: continue

            cv_match = cv_line_pattern.match(line_stripped)
            if cv_match:
                if pending_cv: print(f"  > WARNING: Found verse marker '{line_stripped}' at L{line_num+1} after previous {pending_cv}. Discarding previous.")
                try: pending_cv = (int(cv_match.group(1)), int(cv_match.group(2)))
                except ValueError: print(f"  > WARNING: Invalid C,V format '{line_stripped}' L{line_num+1}."); pending_cv = None
                last_line_was_cv = True
                continue

            if pending_cv:
                potential_book_header = line_stripped.title(); potential_book_header = re.sub(r'^(\d)\s', r'\1 ', potential_book_header)
                if not last_line_was_cv and potential_book_header.lower() in TANAKH_BOOKS_LOWER:
                     print(f"  > WARNING: Found header '{potential_book_header}' L{line_num+1} expecting text for {cur_book} {pending_cv[0]}:{pending_cv[1]}. Discarding verse.")
                     cur_book = potential_book_header; pending_cv = None; last_line_was_cv = False
                     continue

                text = re.sub(r'\{[SPN]\}', '', line_stripped).strip()
                cleaned_text = re.sub(r'\s+', ' ', text).strip()
                if cleaned_text:
                    chap, verse = pending_cv
                    row = { "id": f"{source}-{cur_book}-{chap}-{verse}", "source": source, "ref": f"{cur_book} {chap}:{verse}", "book": cur_book, "chapter": chap, "verse": verse, "text": cleaned_text }
                    write(out, row); count += 1
                else: print(f"  > WARNING: Empty text line after C,V {pending_cv} L{line_num+1}.")
                pending_cv = None; last_line_was_cv = False
                continue

            potential_book_header = line_stripped.title(); potential_book_header = re.sub(r'^(\d)\s', r'\1 ', potential_book_header)
            if not last_line_was_cv and potential_book_header.lower() in TANAKH_BOOKS_LOWER:
                 cur_book = potential_book_header
                 last_line_was_cv = False
                 continue

            last_line_was_cv = False

    if pending_cv: print(f"  > WARNING: End of file while waiting for text for {cur_book} {pending_cv[0]}:{pending_cv[1]}.")
    if count == 0 and in_scripture: print(f"  > FATAL ERROR: Wrote 0 records despite trigger.")
    elif not in_scripture: print(f"  > FATAL ERROR: Wrote 0 records. Trigger 'genesis' not found.")
    else: print(f"  > Wrote {count} records.")


# --- 4. Parser for corpus/en.yusufali.txt ---
# (Confirmed working with warnings - Needs final check after fixes elsewhere)
def parse_quran(inp: str, source: str, out: TextIO):
    print(f"Using dedicated parser: parse_quran (Gutenberg - Ref Fix v2)...")
    verse_num_re = re.compile(r'^(\d{3})\.(\d{3})$')
    yusuf_ali_line_re = re.compile(r'^Y:\s*(.*)')
    current_sura = 0
    current_verse = 0
    yusuf_ali_text_buffer = ""
    in_scripture = False
    count = 0
    last_processed_marker = (0,0)

    with open(inp, "r", encoding="utf-8", errors="ignore") as f:
        # ... [rest of parse_quran - unchanged] ...
        for line_num, line in enumerate(f): # Added line number
            line_stripped = line.strip()

            if "START OF THE PROJECT GUTENBERG EBOOK" in line:
                in_scripture = True; continue
            if not in_scripture or not line_stripped: continue

            verse_match = verse_num_re.match(line_stripped)
            if verse_match:
                if current_sura > 0 and current_verse > 0 and yusuf_ali_text_buffer and (current_sura, current_verse) != last_processed_marker:
                     cleaned_text = re.sub(r'\s+', ' ', yusuf_ali_text_buffer).strip()
                     row = { "id": f"{source}-{current_sura}-{current_verse}", "source": source, "ref": f"Qur'an {current_sura}:{current_verse}", "book": "Qur'an", "chapter": current_sura, "verse": current_verse, "text": cleaned_text }
                     write(out, row); count += 1
                     last_processed_marker = (current_sura, current_verse)
                     yusuf_ali_text_buffer = ""

                try:
                    new_sura = int(verse_match.group(1)); new_verse = int(verse_match.group(2))
                    if new_sura == current_sura and new_verse != current_verse + 1 and current_verse != 0: print(f"  > WARNING: Non-sequential verse at L{line_num+1}. Expected {current_sura}:{current_verse+1}, got {new_sura}:{new_verse}.")
                    current_sura = new_sura; current_verse = new_verse
                except ValueError: print(f"  > WARNING: Invalid marker format '{line_stripped}' at L{line_num+1}."); current_sura = 0; current_verse = 0
                continue

            if current_sura > 0 and current_verse > 0:
                yusuf_match = yusuf_ali_line_re.match(line_stripped)
                if yusuf_match:
                    if yusuf_ali_text_buffer: print(f"  > WARNING: Overwriting buffer for {current_sura}:{current_verse} at L{line_num+1}")
                    yusuf_ali_text_buffer = yusuf_match.group(1).strip()
                elif yusuf_ali_text_buffer and not line_stripped.startswith(('P:', 'S:')) and not verse_num_re.match(line_stripped) and "---" not in line and "Chapter" not in line and "Revealed At:" not in line:
                     yusuf_ali_text_buffer += " " + line_stripped

    if current_sura > 0 and current_verse > 0 and yusuf_ali_text_buffer and (current_sura, current_verse) != last_processed_marker:
        cleaned_text = re.sub(r'\s+', ' ', yusuf_ali_text_buffer).strip()
        row = { "id": f"{source}-{current_sura}-{current_verse}", "source": source, "ref": f"Qur'an {current_sura}:{current_verse}", "book": "Qur'an", "chapter": current_sura, "verse": current_verse, "text": cleaned_text }
        write(out, row); count += 1

    print(f"  > Wrote {count} records.")

# --- 5. Parser for corpus/bhagavad_gita_as_it_is.txt ---
# (Confirmed working - No changes needed)
def parse_gita(inp: str, source: str, out: TextIO):
    print(f"Using dedicated parser: parse_gita...")
    # ... [rest of parse_gita - unchanged] ...
    text_num_re = re.compile(r'^\s*TEXT(?:S)?\s+([\d\-]+)\s*$')
    translation_re = re.compile(r'^\s*TRANSLATION\s*$')
    purport_re = re.compile(r'^\s*PURPORT\s*$')
    junk_line_re = re.compile(r'^[a-zāīūṛḷṁḥṭḍṇñśṣ\- ]+$', re.IGNORECASE)
    text_num_str = None
    translation_lines = []
    purport_lines = []
    parsing_state = None
    count = 0
    in_scripture = False

    with open(inp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped: continue
            if "Introduction" in line_stripped and len(line_stripped) < 20 and not in_scripture:
                in_scripture = True
                continue
            if not in_scripture: continue
            if "Copyright ©" in line: continue
            if "SYNONYMS" in line_stripped:
                parsing_state = None
                continue

            match_text_num = text_num_re.match(line_stripped)
            if match_text_num:
                if text_num_str and translation_lines:
                    full_text = " ".join(translation_lines) + " " + " ".join(purport_lines)
                    cleaned_text = re.sub(r'\s+', ' ', full_text).strip()
                    first_verse_num = int(text_num_str.split('-')[0])
                    row = { "id": f"{source}-text-{text_num_str}", "source": source, "ref": f"Gita Text {text_num_str}", "book": "Gita", "chapter": 0, "verse": first_verse_num, "text": cleaned_text }
                    write(out, row)
                    count += 1
                text_num_str = match_text_num.group(1)
                translation_lines = []
                purport_lines = []
                parsing_state = None
                continue

            if translation_re.match(line_stripped):
                parsing_state = 'translation'
                continue
            if purport_re.match(line_stripped):
                parsing_state = 'purport'
                continue

            if parsing_state == 'translation' and not junk_line_re.match(line_stripped):
                translation_lines.append(line_stripped)
            elif parsing_state == 'purport' and not junk_line_re.match(line_stripped):
                purport_lines.append(line_stripped)

    if text_num_str and translation_lines:
        full_text = " ".join(translation_lines) + " " + " ".join(purport_lines)
        cleaned_text = re.sub(r'\s+', ' ', full_text).strip()
        first_verse_num = int(text_num_str.split('-')[0])
        row = { "id": f"{source}-text-{text_num_str}", "source": source, "ref": f"Gita Text {text_num_str}", "book": "Gita", "chapter": 0, "verse": first_verse_num, "text": cleaned_text }
        write(out, row)
        count += 1
    print(f"  > Wrote {count} records.")

# --- 6. Parser for corpus/damapada.txt (ASCII ID Fix) ---
def parse_dhammapada(inp: str, source: str, out: TextIO):
    print(f"Using dedicated parser: parse_dhammapada (ASCII ID Fix)...")
    chapter_re = re.compile(r'^\s*(I|V|X|L|C|D|M)+\s*\:\s*([\w\s()]+)\s*$') # Allow more roman numerals & ()
    verse_re = re.compile(r'^\s*([\d–-]+)\s*\*?$') # Allows ranges like 1-2, 3–6, single like 25
    cur_chapter_name = "Unknown" # Default chapter name
    verse_num_str_raw = None # The raw verse string, e.g., "1-2" or "3–6" or "25"
    verse_lines = []
    count = 0
    in_scripture = False

    with open(inp, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f): # Added line number
            line_stripped = line.strip()

            # State trigger: Look for first chapter marker more reliably
            # Example: "I : Pairs" or "II : Heedfulness"
            if not in_scripture:
                 match_chap = chapter_re.match(line_stripped)
                 if match_chap:
                     in_scripture = True
                     cur_chapter_name = match_chap.group(2).strip()
                     # print(f"DEBUG: Triggered scripture start L{line_num+1}, Chapter: {cur_chapter_name}")
                     continue # Skip the trigger line itself
                 else:
                     continue # Keep skipping intro

            if not line_stripped: continue # Skip blank lines within scripture

            # Found a new chapter
            match_chap = chapter_re.match(line_stripped)
            if match_chap:
                # Write the last verse of the previous chapter
                if verse_num_str_raw and verse_lines:
                    cleaned_text = re.sub(r'\s+', ' ', " ".join(verse_lines)).strip()
                    # Use ASCII hyphen consistently for ID and Ref
                    verse_num_str_clean = verse_num_str_raw.replace('–', '-')
                    first_verse_num = int(re.split(r'[-–]', verse_num_str_raw)[0]) # Get first num even with en dash
                    row = {
                        "id": f"{source}-{cur_chapter_name.replace(' ','_')}-{verse_num_str_clean}", # Use clean version for ID
                        "source": source,
                        "ref": f"Dhammapada {cur_chapter_name} {verse_num_str_clean}", # Use clean version for Ref
                        "book": "Dhammapada", "chapter": 0, # Chapter isn't numbered consistently
                        "verse": first_verse_num, "text": cleaned_text
                    }
                    write(out, row); count += 1
                cur_chapter_name = match_chap.group(2).strip()
                verse_num_str_raw = None; verse_lines = []
                # print(f"DEBUG: Found Chapter: {cur_chapter_name} L{line_num+1}")
                continue

            # Found a new verse number (or range)
            match_verse = verse_re.match(line_stripped)
            if match_verse:
                # Write the previous verse
                if verse_num_str_raw and verse_lines:
                    cleaned_text = re.sub(r'\s+', ' ', " ".join(verse_lines)).strip()
                    verse_num_str_clean = verse_num_str_raw.replace('–', '-')
                    first_verse_num = int(re.split(r'[-–]', verse_num_str_raw)[0])
                    row = {
                        "id": f"{source}-{cur_chapter_name.replace(' ','_')}-{verse_num_str_clean}",
                        "source": source, "ref": f"Dhammapada {cur_chapter_name} {verse_num_str_clean}",
                        "book": "Dhammapada", "chapter": 0,
                        "verse": first_verse_num, "text": cleaned_text
                    }
                    write(out, row); count += 1

                # Start new verse - store the raw string with potential en dash
                verse_num_str_raw = match_verse.group(1)
                verse_lines = []
                continue # Skip the number line itself

            # This is a line of text for the current verse
            if verse_num_str_raw and cur_chapter_name:
                verse_lines.append(line_stripped)

    # Write the very last verse after the loop ends
    if verse_num_str_raw and verse_lines:
        cleaned_text = re.sub(r'\s+', ' ', " ".join(verse_lines)).strip()
        verse_num_str_clean = verse_num_str_raw.replace('–', '-')
        first_verse_num = int(re.split(r'[-–]', verse_num_str_raw)[0])
        row = {
            "id": f"{source}-{cur_chapter_name.replace(' ','_')}-{verse_num_str_clean}",
            "source": source, "ref": f"Dhammapada {cur_chapter_name} {verse_num_str_clean}",
            "book": "Dhammapada", "chapter": 0,
            "verse": first_verse_num, "text": cleaned_text
        }
        write(out, row); count += 1
    print(f"  > Wrote {count} records.")


# ---------- CLI Dispatcher ----------
# (No changes needed)
def main():
    ap = argparse.ArgumentParser(description="Chunk scriptures to JSONL using dedicated parsers for each format.")
    ap.add_argument("source", choices=["quran", "bible_asv", "bible_nrsv", "tanakh", "gita", "dhammapada"], help="Specify the source text type.")
    ap.add_argument("inp", help="Path to the input .txt file.")
    ap.add_argument("out", help="Path to the output .jsonl file.")
    args = ap.parse_args()

    print(f"Processing '{args.inp}' -> '{args.out}'...")

    parser_map = {
        "bible_asv": parse_asv_bible,
        "bible_nrsv": parse_nrsv_bible,
        "tanakh": parse_tanakh,
        "quran": parse_quran,
        "gita": parse_gita,
        "dhammapada": parse_dhammapada
    }

    if args.source in parser_map:
        try: # Add top-level try block for the parser call
            with open(args.out, "w", encoding="utf-8") as outfile:
                parser_map[args.source](args.inp, args.source, outfile)
        except Exception as e:
            print(f"  > FATAL ERROR during execution of parser for {args.source}: {e}")
            sys.exit(1) # Exit if the parser itself crashes
    else:
        # This case should not be reachable due to argparse choices
        print(f"Error: Unknown source '{args.source}'. No dedicated parser available.")
        sys.exit(1)

    print(f"Finished processing '{args.inp}'.")

if __name__ == "__main__":
    main()
