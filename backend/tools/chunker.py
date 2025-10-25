# tools/chunker.py
# Final Correction: Fixed Dhammapada parser to use ASCII hyphen in IDs.
# REVISION 4: Rewrote 'parse_nrsv_bible' AGAIN. v3 failed on "I am".
# REVISION 2: Rewrote 'parse_tanakh' to fix verse-merging bugs.
import sys
import json
import re
import argparse
import os
from typing import Dict, TextIO

# --- Data for Reference Checks ---
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
def write(out: TextIO, row: Dict):
    """Writes a dictionary as a JSON line to the output file."""
    out.write(json.dumps(row, ensure_ascii=False) + "\n")

# ---------- 6 DEDICATED PARSERS ----------

# --- 1. Parser for corpus/ASVHB.txt ---
# (Confirmed working - No changes needed)
def parse_asv_bible(inp: str, source: str, out: TextIO):
    print(f"Using dedicated parser: parse_asv_bible...")
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
# (START OF REWRITTEN SECTION v4)

# Helper function to parse chapter/verse numbers (int or roman)
def _parse_roman_or_int(num_str: str) -> int:
    """Parses a string that is either an integer or a Roman numeral."""
    num_str = num_str.strip().lower()
    if num_str.isdigit():
        try:
            return int(num_str)
        except ValueError:
            return 0
    
    roman_map = {'i': 1, 'v': 5, 'x': 10, 'l': 50, 'c': 100, 'd': 500, 'm': 1000}
    val = 0
    prev_val = 0
    for n in reversed(num_str):
        if n not in roman_map:
            return 0 # Invalid roman numeral
        curr = roman_map[n]
        val += curr if curr >= prev_val else -curr
        prev_val = curr
    return val

# Helper function to write the buffered verse
def _write_nrsv_verse(out: TextIO, source: str, book: str, chap: int, verse_num: int, text_buffer: str) -> int:
    """Writes the buffered verse text to the output file if valid."""
    if book and chap > 0 and verse_num > 0 and text_buffer:
        cleaned_text = re.sub(r'\s+', ' ', text_buffer).strip()
        if cleaned_text:
            row = {
                "id": f"{source}-{book.replace(' ', '_')}-{chap}-{verse_num}",
                "source": source,
                "ref": f"{book} {chap}:{verse_num}",
                "book": book,
                "chapter": chap,
                "verse": verse_num,
                "text": cleaned_text
            }
            write(out, row) # Use the global 'write' function
            return 1
    return 0


def parse_nrsv_bible(inp: str, source: str, out: TextIO):
    print(f"Using dedicated parser: parse_nrsv_bible (REWRITTEN v4 - Dual-Regex)...")
    cur_book = ""
    cur_chapter = 0
    cur_verse_num = 0
    current_text_buffer = ""
    count = 0

    chapter_marker_re = re.compile(r'^\s*\[([A-Za-z\s.-]+?)\s+([0-9ivxlcdm]+)\]\s*$', re.IGNORECASE)
    # Regex for verse lines *starting* with a number (digits OR roman)
    verse_line_re = re.compile(r'^\s*([ivxlcdm\d]+)\s+(.*)', re.IGNORECASE)
    # Regex for *in-line* verse numbers (DIGITS ONLY)
    inline_verse_marker_re = re.compile(r'\s(\d+)\s+')
    # Regex for page headers/footers (e.g., "GENESIS 32" or "31")
    page_junk_re = re.compile(r'^\s*(\d+\s+[A-Z\s]+|[A-Z\s]+\s+\d+)\s*$|^\s*\d+\s*$')

    try:
        with open(inp, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f):
                line_stripped = line.strip()
                if not line_stripped:
                    continue

                # 1. Check for Chapter Marker
                chap_match = chapter_marker_re.match(line_stripped)
                if chap_match:
                    count += _write_nrsv_verse(out, source, cur_book, cur_chapter, cur_verse_num, current_text_buffer)
                    book_name, chap_str = chap_match.groups()
                    cur_book = book_name.strip().title()
                    cur_book = re.sub(r'^(\d)\s', r'\1 ', cur_book) # Fix "1 Samuel"
                    cur_chapter = _parse_roman_or_int(chap_str)
                    cur_verse_num = 0
                    current_text_buffer = ""
                    if cur_chapter == 0:
                         print(f"> WARN: Bad chap '{chap_str}' L{line_num+1}")
                    continue

                # Skip lines until first chapter is found
                if not cur_book or cur_chapter == 0:
                    continue 

                # 2. Skip page headers/footers
                if page_junk_re.match(line_stripped):
                    continue

                line_to_process = line_stripped
                new_verse_started_on_line = False

                # 3. Check for Line-Starting Verse (like 'i In the beginning...')
                verse_line_match = verse_line_re.match(line_stripped)
                if verse_line_match:
                    verse_num_str, rest_of_line = verse_line_match.groups()
                    new_verse_num = _parse_roman_or_int(verse_num_str)
                    
                    if new_verse_num == 0: # Bad parse, treat as text
                         if cur_verse_num > 0: current_text_buffer += " " + line_stripped
                         continue # Skip rest of processing
                    
                    # This is a valid line-starting verse
                    count += _write_nrsv_verse(out, source, cur_book, cur_chapter, cur_verse_num, current_text_buffer)
                    cur_verse_num = new_verse_num
                    current_text_buffer = "" # Reset buffer
                    line_to_process = rest_of_line # Only process the rest of the line
                    new_verse_started_on_line = True
                
                # 4. Process the line (or rest_of_line) for *inline digit* markers
                line_cursor = 0
                matches = list(inline_verse_marker_re.finditer(line_to_process))

                if not matches:
                    # No inline markers, this is a continuation line (or first part)
                    if cur_verse_num > 0:
                        current_text_buffer += " " + line_to_process
                    # else: this is commentary, like "The primeval history..."
                    continue
                
                # We have inline digit markers, process them
                for match in matches:
                    # 1. Get text *before* this marker
                    text_slice = line_to_process[line_cursor:match.start()]
                    
                    if cur_verse_num > 0:
                        current_text_buffer += " " + text_slice.strip()
                    
                    # If this is the *first* verse marker on the line (e.g. '2' in 'i...2...3')
                    # AND a verse was NOT already started (e.g. 'i'),
                    # then we write the buffer.
                    if not new_verse_started_on_line or line_cursor > 0:
                         count += _write_nrsv_verse(out, source, cur_book, cur_chapter, cur_verse_num, current_text_buffer)

                    # 4. Start the *new* verse
                    verse_num_str = match.group(1)
                    cur_verse_num = int(verse_num_str) # Safe, regex is digits only
                    current_text_buffer = "" # Reset buffer for the new verse
                    line_cursor = match.end() # Move cursor past the marker

                # 5. Add any remaining text *after* the last marker
                remaining_text = line_to_process[line_cursor:].strip()
                if cur_verse_num > 0:
                    current_text_buffer += " " + remaining_text

    except Exception as e:
        print(f"  > FATAL ERROR during file processing: {e} on line {line_num+1}")

    # Write the very last verse after the loop ends
    count += _write_nrsv_verse(out, source, cur_book, cur_chapter, cur_verse_num, current_text_buffer)

    if count == 0:
        print(f"  > FATAL ERROR: Wrote 0 records.")
    else:
        print(f"  > Wrote {count} records.")

# (END OF REWRITTEN SECTION v4)


# --- 3. Parser for corpus/tanakh_et.txt ---
# (REWRITTEN v2 - Confirmed Working)
def parse_tanakh(inp: str, source: str, out: TextIO):
    print(f"Using dedicated parser: parse_tanakh (REWRITTEN for multiline)...")
    cur_book = ""
    cur_chap = 0
    cur_verse = 0
    current_text_buffer = ""
    in_scripture = False
    count = 0
    
    cv_line_pattern = re.compile(r'^\s*(\d+),(\d+)\s*$')
    
    # Helper to write the buffer
    def _write_verse(book, chap, verse, text):
        if book and chap > 0 and verse > 0 and text:
            cleaned_text = re.sub(r'\s+', ' ', text).strip()
            cleaned_text = re.sub(r'\{[SPN]\}', '', cleaned_text).strip() # Clean tags here
            if cleaned_text:
                row = { "id": f"{source}-{book.replace(' ', '_')}-{chap}-{verse}", "source": source, "ref": f"{book} {chap}:{verse}", "book": book, "chapter": chap, "verse": verse, "text": cleaned_text }
                write(out, row)
                return 1
        return 0

    try:
        with open(inp, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f):
                line_stripped = line.strip()
                if not line_stripped:
                    continue

                if not in_scripture:
                    if line_stripped.lower() == "genesis":
                        in_scripture = True
                        cur_book = "Genesis"
                    continue

                cv_match = cv_line_pattern.match(line_stripped)
                potential_book_header_raw = line_stripped.title()
                potential_book_header = re.sub(r'^(\d)\s', r'\1 ', potential_book_header_raw)
                is_book_header = potential_book_header.lower() in TANAKH_BOOKS_LOWER

                if cv_match:
                    # Found a new verse marker. Write the previous one.
                    count += _write_verse(cur_book, cur_chap, cur_verse, current_text_buffer)
                    
                    # Start the new verse
                    try:
                        cur_chap = int(cv_match.group(1))
                        cur_verse = int(cv_match.group(2))
                        current_text_buffer = "" # Reset buffer
                    except ValueError:
                        print(f"  > WARNING: Invalid C,V format '{line_stripped}' L{line_num+1}.")
                        cur_chap, cur_verse = 0, 0 # Invalidate
                    continue

                if is_book_header:
                    # Found a new book header. Write the previous verse.
                    count += _write_verse(cur_book, cur_chap, cur_verse, current_text_buffer)
                    
                    # Start the new book
                    cur_book = potential_book_header
                    cur_chap, cur_verse = 0, 0
                    current_text_buffer = ""
                    continue

                # If it's not a marker, and we are in a valid verse, append text.
                if cur_book and cur_chap > 0 and cur_verse > 0:
                    # This is a continuation line of verse text
                    cleaned_line = re.sub(r'\{[SPN]\}', '', line_stripped).strip()
                    if cleaned_line:
                        current_text_buffer += " " + cleaned_line
                # else:
                    # This skips lines between books/before first verse, which is good.

    except Exception as e:
        print(f"  > FATAL ERROR during file processing: {e} on line {line_num+1}")

    # Write the very last verse
    count += _write_verse(cur_book, cur_chap, cur_verse, current_text_buffer)

    if count == 0:
        print(f"  > FATAL ERROR: Wrote 0 records.")
    else:
        print(f"  > Wrote {count} records.")

# (END OF REWRITTEN SECTION v2)


# --- 4. Parser for corpus/en.yusufali.txt ---
# (Confirmed working - No changes needed)
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
# (Confirmed working - No changes needed)
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