# tools/chunker.py
import sys
import json
import re
import argparse
import traceback
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


# --- Utility Functions ---
def write(out: TextIO, row: Dict):
    """Writes a dictionary as a JSON line to the output file."""
    out.write(json.dumps(row, ensure_ascii=False) + "\n")

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

def roman_word_to_int(roman_word):
    """Converts Roman numeral words (One to Eighteen) to integers."""
    mapping = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18
    }
    return mapping.get(roman_word.lower(), 0) # Return 0 if not found


# ---------- 6 DEDICATED PARSERS ----------

# --- 1. Parser for corpus/ASVHB.txt ---

def _write_asv_verse(out: TextIO, source: str, ref_parts: tuple, text: str) -> int:
    """Helper for parse_asv_bible to write a single verse."""
    if not ref_parts:
        return 0
    book, chap, verse = ref_parts
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    if cleaned_text:
        row = { "id": f"{source}-{book}-{chap}-{verse}", "source": source, "ref": f"{book} {chap}:{verse}", "book": book, "chapter": chap, "verse": verse, "text": cleaned_text }
        write(out, row)
        return 1
    return 0

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
                count += _write_asv_verse(out, source, current_ref_parts, current_text)
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

                count += _write_asv_verse(out, source, current_ref_parts, current_text)

                chap, verse = int(match.group(1)), int(match.group(2))
                current_ref_parts = (cur_book, chap, verse)
                current_text = ""
                line_cursor = match.end()

            remaining_text = line[line_cursor:].strip()
            if current_text is not None: current_text += " " + remaining_text
            else: current_text = remaining_text

    count += _write_asv_verse(out, source, current_ref_parts, current_text)
    print(f"   > Wrote {count} records.")


# --- 2. Parser for corpus/nrsv_bible.txt ---

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
    print(f"Using dedicated parser: parse_nrsv_bible...")
    cur_book = ""
    cur_chapter = 0
    cur_verse_num = 0
    current_text_buffer = ""
    count = 0

    chapter_marker_re = re.compile(r'^\s*\[([A-Za-z\s.-]+?)\s+([0-9ivxlcdm]+)\]\s*$', re.IGNORECASE)
    verse_line_re = re.compile(r'^\s*([ivxlcdm\d]+)\s+(.*)', re.IGNORECASE)
    inline_verse_marker_re = re.compile(r'\s(\d+)\s+')
    page_junk_re = re.compile(r'^\s*(\d+\s+[A-Z\s]+|[A-Z\s]+\s+\d+)\s*$|^\s*\d+\s*$')

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

            # 3. Check for Line-Starting Verse
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
                # No inline markers, this is a continuation line
                if cur_verse_num > 0:
                    current_text_buffer += " " + line_to_process
                continue
            
            # We have inline digit markers, process them
            for match in matches:
                # 1. Get text *before* this marker
                text_slice = line_to_process[line_cursor:match.start()]
                
                # Append this slice to the current buffer
                if cur_verse_num > 0:
                    current_text_buffer += " " + text_slice.strip()
                
                # Write the completed previous verse before starting new one
                is_first_marker = (line_cursor == 0)
                if not (new_verse_started_on_line and is_first_marker):
                        count += _write_nrsv_verse(out, source, cur_book, cur_chapter, cur_verse_num, current_text_buffer)

                # 4. Start the *new* verse
                verse_num_str = match.group(1)
                cur_verse_num = int(verse_num_str)
                current_text_buffer = "" # Reset buffer
                line_cursor = match.end()

            # 5. Add any remaining text *after* the last marker
            remaining_text = line_to_process[line_cursor:].strip()
            if cur_verse_num > 0:
                current_text_buffer += " " + remaining_text

    # Write the very last verse after the loop ends
    count += _write_nrsv_verse(out, source, cur_book, cur_chapter, cur_verse_num, current_text_buffer)

    if count == 0:
        print(f"   > FATAL ERROR: Wrote 0 records.")
    else:
        print(f"   > Wrote {count} records.")


# --- 3. Parser for corpus/tanakh_et.txt ---

def _write_tanakh_verse(out: TextIO, source: str, book: str, chap: int, verse: int, text: str) -> int:
    """Helper for parse_tanakh to write a single verse."""
    if book and chap > 0 and verse > 0 and text:
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        cleaned_text = re.sub(r'\{[SPN]\}', '', cleaned_text).strip() # Clean tags
        if cleaned_text:
            row = { "id": f"{source}-{book.replace(' ', '_')}-{chap}-{verse}", "source": source, "ref": f"{book} {chap}:{verse}", "book": book, "chapter": chap, "verse": verse, "text": cleaned_text }
            write(out, row)
            return 1
    return 0

def parse_tanakh(inp: str, source: str, out: TextIO):
    print(f"Using dedicated parser: parse_tanakh...")
    cur_book = ""
    cur_chap = 0
    cur_verse = 0
    current_text_buffer = ""
    in_scripture = False
    count = 0
    
    cv_line_pattern = re.compile(r'^\s*(\d+),(\d+)\s*$')

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
                count += _write_tanakh_verse(out, source, cur_book, cur_chap, cur_verse, current_text_buffer)
                
                # Start the new verse
                try:
                    cur_chap = int(cv_match.group(1))
                    cur_verse = int(cv_match.group(2))
                    current_text_buffer = "" # Reset buffer
                except ValueError:
                    print(f"   > WARNING: Invalid C,V format '{line_stripped}' L{line_num+1}.")
                    cur_chap, cur_verse = 0, 0 # Invalidate
                continue

            if is_book_header:
                # Found a new book header. Write the previous verse.
                count += _write_tanakh_verse(out, source, cur_book, cur_chap, cur_verse, current_text_buffer)
                
                # Start the new book
                cur_book = potential_book_header
                cur_chap, cur_verse = 0, 0
                current_text_buffer = ""
                continue

            # If it's not a marker, and we are in a valid verse, append text.
            if cur_book and cur_chap > 0 and cur_verse > 0:
                cleaned_line = re.sub(r'\{[SPN]\}', '', line_stripped).strip()
                if cleaned_line:
                    current_text_buffer += " " + cleaned_line

    # Write the very last verse
    count += _write_tanakh_verse(out, source, cur_book, cur_chap, cur_verse, current_text_buffer)

    if count == 0:
        print(f"   > FATAL ERROR: Wrote 0 records.")
    else:
        print(f"   > Wrote {count} records.")


# --- 4. Parser for corpus/en.yusufali.txt ---
# (REPLACED with v11 - Two-Pass Multi-line Name Fix + Parenthesis Skip)

def _write_quran_verse(out: TextIO, source: str, sura: int, verse: int, text: str, sura_name: str) -> int:
    """Helper for parse_quran to write a single verse."""
    if sura > 0 and verse > 0 and text:
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        
        # Use Surah name in ref string if it exists
        ref_name = sura_name if sura_name else f"{sura}"
        ref_str = f"Qur'an {ref_name} {sura}:{verse}".strip().replace("  ", " ")

        row = {
            "id": f"{source}-{sura}-{verse}",
            "source": source,
            "ref": ref_str,
            "book": "Qur'an",
            "chapter": sura,
            "verse": verse,
            "text": cleaned_text
        }
        write(out, row)
        return 1
    return 0

def parse_quran(inp: str, source: str, out: TextIO):
    print(f"Using dedicated parser: parse_quran (v11 - Multi-line Name Fix)...")
    
    # --- Regexes ---
    chapter_num_re = re.compile(r'^\s*(?:Chapter|Surah)\s+(\d+):?\s*$', re.IGNORECASE)
    verse_num_re = re.compile(r'^(\d{3})\.(\d{3})$')
    yusuf_ali_line_re = re.compile(r'^Y:\s*(.*)')

    SURAH_NAMES = {}
    
    # --- PASS 1: Build the name cache ---
    print("   > Pass 1: Scanning for Surah names...")
    expecting_name = False
    current_sura_num_for_cache = 0
    
    try:
        with open(inp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_stripped = line.strip()

                if not line_stripped:
                    continue 

                # State 1: Look for "Chapter [num]:"
                num_match = chapter_num_re.match(line_stripped)
                if num_match:
                    current_sura_num_for_cache = int(num_match.group(1))
                    expecting_name = True
                    continue

                # State 2: We are expecting a name. This line *must* be it.
                if expecting_name:
                    if "Total Verses:" in line or "Revealed At:" in line or "---" in line:
                        continue
                        
                    sura_name_full = line_stripped # e.g., "AL-FATIHA (THE OPENING)"
                    # --- MODIFICATION: Remove parenthesis ---
                    sura_name = sura_name_full.split('(', 1)[0].strip()
                    # --- END MODIFICATION ---
                    SURAH_NAMES[current_sura_num_for_cache] = sura_name
                    expecting_name = False 
                    current_sura_num_for_cache = 0
                    
    except Exception as e:
        print(f"   > FATAL ERROR during Pass 1 (Name Scan): {e}")
        return

    print(f"   > Pass 1: Found {len(SURAH_NAMES)} Surah names.")
    if not SURAH_NAMES:
        print("   > FATAL: No Surah names found. Check file format.")
        return

    # --- PASS 2: Parse verses ---
    print("   > Pass 2: Parsing verses...")
    current_sura = 0
    current_verse = 0
    current_text = ""
    current_sura_name = ""
    in_scripture = False
    count = 0

    try:
        with open(inp, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f):
                line_stripped = line.strip()

                if not in_scripture:
                    if "001.001" in line: # Start parsing from the first verse marker
                        in_scripture = True
                    else:
                        continue
                
                if not line_stripped: continue

                # Check for Verse marker
                verse_match = verse_num_re.match(line_stripped)
                if verse_match:
                    # Write the previously accumulated verse data
                    count += _write_quran_verse(out, source, current_sura, current_verse, current_text, current_sura_name)

                    # Start the NEW verse
                    try:
                        new_sura = int(verse_match.group(1))
                        new_verse = int(verse_match.group(2))
                        
                        current_sura = new_sura
                        current_verse = new_verse
                        current_text = "" # Reset text buffer
                        
                        # Get the name from our pre-built cache
                        current_sura_name = SURAH_NAMES.get(current_sura, "")
                        
                    except ValueError:
                        print(f"   > WARNING (Pass 2): Invalid marker format '{line_stripped}' at L{line_num+1}.")
                        current_sura = 0; current_verse = 0 # Invalidate state
                    continue

                # Process Yusuf Ali text lines ('Y: ...') or continuations
                if current_sura > 0 and current_verse > 0: # Only if we are in a valid verse
                    yusuf_match = yusuf_ali_line_re.match(line_stripped)
                    if yusuf_match:
                        current_text += " " + yusuf_match.group(1).strip()
                    # Handle continuation lines
                    elif current_text and not line_stripped.startswith(('P:', 'S:')) and "---" not in line and "Revealed At:" not in line and not chapter_num_re.match(line_stripped):
                            current_text += " " + line_stripped

    except Exception as e:
        print(f"   > FATAL ERROR during Pass 2 (Verse Parse): {e} on line {line_num+1 if 'line_num' in locals() else 'unknown'}")

    # Write the very last verse buffer after the loop ends
    count += _write_quran_verse(out, source, current_sura, current_verse, current_text, current_sura_name)

    if count == 0:
         print(f"   > FATAL ERROR: Wrote 0 records.")
    else:
        print(f"   > Wrote {count} records.")


# --- 5. Parser for corpus/bhagavad_gita_as_it_is.txt ---
# (Using the v8 version you confirmed was working)

def _write_gita_text(out: TextIO, source: str, chap_num: int, chap_name: str, text_num_str: str, lines: list) -> int:
    """Helper for parse_gita to write a single text block."""
    if text_num_str and lines and chap_num > 0:
        full_text = " ".join(lines)
        cleaned_text = re.sub(r'\s+', ' ', full_text).strip()
        try:
            first_verse_num = int(text_num_str.split('-')[0])
        except ValueError:
            first_verse_num = 0

        if cleaned_text:
            # Use the chapter name in the ref if we have it
            ref_chapter_display = f"{chap_name.strip()} {chap_num}" if chap_name.strip() else f"Chapter {chap_num}"

            row = {
                "id": f"{source}-Ch{chap_num}-Text{text_num_str}", # Keep ID stable
                "source": source,
                "ref": f"Gita {ref_chapter_display}: Text {text_num_str}", # Use the name
                "book": "Gita",
                "chapter": chap_num,
                "verse": first_verse_num,
                "text": cleaned_text
            }
            write(out, row)
            return 1
    return 0

def parse_gita(inp: str, source: str, out: TextIO):
    print(f"Using dedicated parser: parse_gita (v8 - Chapter/Copyright Fix)...")
    
    # --- Regexes ---
    chapter_re = re.compile(r'^\s*[-]*\s*CHAPTER\s+([\w\d]+)\s*[-]*\s*$', re.IGNORECASE)
    text_num_re = re.compile(r'^\s*TEXT(?:S)?\s+([\d\-]+)\s*$')
    translation_re = re.compile(r'^\s*TRANSLATION\s*$')
    purport_re = re.compile(r'^\s*PURPORT\s*$')
    sanskrit_junk_re = re.compile(r'^[a-zāīūṛḷṁḥṭḍṇñśṣ\- ]+$', re.IGNORECASE)

    # --- State variables ---
    current_chapter_num = 0
    current_chapter_name = ""
    current_text_num_str = None
    current_translation_lines = []
    
    parsing_translation = False
    expecting_chapter_title = False # New flag to grab the title from the next line
    
    count = 0
    in_scripture = False # Flag to skip intro

    with open(inp, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f):
            line_stripped = line.strip()
            if not line_stripped: continue

            if "copyright" in line.lower():
                continue

            if not in_scripture:
                chap_match = chapter_re.match(line_stripped)
                if chap_match:
                    in_scripture = True
                else:
                    continue 

            # --- Process lines after intro ---
            chap_match = chapter_re.match(line_stripped)
            if chap_match:
                count += _write_gita_text(out, source, current_chapter_num, current_chapter_name, current_text_num_str, current_translation_lines)
                current_text_num_str = None
                current_translation_lines = []
                parsing_translation = False
                
                num_str = chap_match.group(1)
                if num_str.isdigit():
                    current_chapter_num = int(num_str)
                else:
                    current_chapter_num = roman_word_to_int(num_str)
                
                current_chapter_name = "" 
                expecting_chapter_title = True 
                continue

            if expecting_chapter_title and not text_num_re.match(line_stripped):
                current_chapter_name = line_stripped
                expecting_chapter_title = False
                continue

            match_text_num = text_num_re.match(line_stripped)
            if match_text_num:
                count += _write_gita_text(out, source, current_chapter_num, current_chapter_name, current_text_num_str, current_translation_lines)
                current_text_num_str = match_text_num.group(1)
                current_translation_lines = []
                parsing_translation = False 
                expecting_chapter_title = False 
                continue

            if translation_re.match(line_stripped):
                parsing_translation = True
                continue

            if purport_re.match(line_stripped) or "SYNONYMS" in line_stripped:
                parsing_translation = False
                continue

            if parsing_translation and current_chapter_num > 0 and current_text_num_str:
                if not sanskrit_junk_re.match(line_stripped):
                    current_translation_lines.append(line_stripped)

    # Write the very last text after the loop ends
    count += _write_gita_text(out, source, current_chapter_num, current_chapter_name, current_text_num_str, current_translation_lines)

    if count == 0:
        print(f"   > FATAL ERROR: Wrote 0 records.")
    else:
        print(f"   > Wrote {count} records.")


# --- 6. Parser for corpus/damapada.txt ---

def _write_dhammapada_verse(out: TextIO, source: str, chapter_name: str, verse_num_raw: str, lines: list) -> int:
    """Helper for parse_dhammapada to write a single verse."""
    if not verse_num_raw or not lines:
        return 0
    
    cleaned_text = re.sub(r'\s+', ' ', " ".join(lines)).strip()
    if not cleaned_text:
        return 0

    verse_num_str_clean = verse_num_raw.replace('–', '-')
    try:
        first_verse_num = int(re.split(r'[-–]', verse_num_raw)[0])
    except (ValueError, IndexError):
        first_verse_num = 0 # Safety

    row = {
        "id": f"{source}-{chapter_name.replace(' ','_')}-{verse_num_str_clean}",
        "source": source,
        "ref": f"Dhammapada {chapter_name} {verse_num_str_clean}",
        "book": "Dhammapada", "chapter": 0,
        "verse": first_verse_num, "text": cleaned_text
    }
    write(out, row)
    return 1

def parse_dhammapada(inp: str, source: str, out: TextIO):
    print(f"Using dedicated parser: parse_dhammapada...")
    chapter_re = re.compile(r'^\s*(I|V|X|L|C|D|M)+\s*\:\s*([\w\s()]+)\s*$')
    verse_re = re.compile(r'^\s*([\d–-]+)\s*\*?$')
    cur_chapter_name = "Unknown"
    verse_num_str_raw = None
    verse_lines = []
    count = 0
    in_scripture = False

    with open(inp, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f):
            line_stripped = line.strip()

            if not in_scripture:
                match_chap = chapter_re.match(line_stripped)
                if match_chap:
                    in_scripture = True
                    cur_chapter_name = match_chap.group(2).strip()
                    continue
                else:
                    continue

            if not line_stripped: continue

            match_chap = chapter_re.match(line_stripped)
            if match_chap:
                count += _write_dhammapada_verse(out, source, cur_chapter_name, verse_num_str_raw, verse_lines)
                cur_chapter_name = match_chap.group(2).strip()
                verse_num_str_raw = None; verse_lines = []
                continue

            match_verse = verse_re.match(line_stripped)
            if match_verse:
                count += _write_dhammapada_verse(out, source, cur_chapter_name, verse_num_str_raw, verse_lines)
                verse_num_str_raw = match_verse.group(1)
                verse_lines = []
                continue

            if verse_num_str_raw and cur_chapter_name:
                verse_lines.append(line_stripped)

    # Write the very last verse after the loop ends
    count += _write_dhammapada_verse(out, source, cur_chapter_name, verse_num_str_raw, verse_lines)
    print(f"   > Wrote {count} records.")


# ---------- CLI Dispatcher ----------
def main():
    ap = argparse.ArgumentParser(description="Chunk scriptures to JSONL using dedicated parsers for each format.")
    ap.add_argument("source", choices=["quran", "bible_asv", "bible_nrsv", "tanakh", "gita", "dhammapada"], help="Specify the source text type.")
    ap.add_argument("inp", help="Path to the input .txt file.") # <-- FIXED TYPO HERE
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
        try:
            with open(args.out, "w", encoding="utf-8") as outfile:
                parser_map[args.source](args.inp, args.source, outfile)
        except Exception as e:
            print(f"   > FATAL ERROR during execution of parser for {args.source}: {e}")
            traceback.print_exc()
            sys.exit(1)

    print(f"Finished processing '{args.inp}'.")

if __name__ == "__main__":
    main()