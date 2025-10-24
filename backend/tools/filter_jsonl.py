# tools/filter_jsonl.py
# Final Correction: Added 'from typing import Dict' to fix NameError.
# Reads an _embed.jsonl file, applies data quality checks,
# and writes only the valid lines to a new _clean_embed.jsonl file.
import sys
import json
import re
import argparse
import os # Import os for path checks
from typing import Dict # <--- IMPORT ADDED HERE

# --- Data for Reference Checks ---
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
# Note: 'out' parameter no longer needs type hint if TextIO isn't imported
def write(out, row: Dict):
    """Writes a dictionary as a JSON line to the output file."""
    out.write(json.dumps(row, ensure_ascii=False) + "\n")

# --- Filter Logic ---
def is_suspicious_reference(ref_str: str) -> bool:
    """Checks if a reference string seems invalid (e.g., impossible chapter/verse)."""
    if not ref_str or ':' not in ref_str: return False # Basic format check

    # Quran Check
    if ref_str.lower().startswith("qur'an"):
        parts = ref_str.split(':')
        if len(parts) == 2:
            try:
                sura_part = parts[0].split()[-1]; sura = int(sura_part)
                verse = int(parts[1])
                if sura <= 0 or sura > QURAN_CHAPTER_LIMITS or verse <= 0 or verse > 300: return True
            except (ValueError, IndexError): return True
        else: return True
        return False

    # Bible/Tanakh Check
    # Match "Book Name Chapter:Verse" - allows for spaces and digits in book name
    match = re.match(r'^([1-3]?\s?[A-Za-z\s.-]+?)\s+(\d+):(\d+)$', ref_str)
    if match:
        book, chap_str, verse_str = match.groups()
        book = book.strip().title(); book = re.sub(r'^(\d)\s', r'\1 ', book) # Normalize "1 Samuel"

        try:
            chap = int(chap_str); verse = int(verse_str)
            limit = -1
            book_check_lower = book.lower().replace('i chronicles', '1 chronicles').replace('ii chronicles', '2 chronicles')
            # Handle Song of Solomon/Songs alias
            if book_check_lower == "song of solomon": book_check_lower = "song of songs"
            if book_check_lower == "ecclesiasticus": book_check_lower = "sirach" # alias Sirach
            if book_check_lower == "wisdom of solomon": book_check_lower = "wisdom" # alias Wisdom


            for known_book, chap_limit in BIBLE_CHAPTER_LIMITS.items():
                if book_check_lower == known_book.lower():
                    limit = chap_limit
                    break

            if limit == -1: return True # Book not recognized
            # Increased verse limit slightly for edge cases like Psalms
            if chap <= 0 or chap > limit or verse <= 0 or verse > 250: return True
        except ValueError: return True
    else:
        # Allow Gita/Dhammapada refs
        if "Gita Text" in ref_str or "Dhammapada" in ref_str: return False
        return True # Doesn't match expected Bible or Quran format

    return False

def is_short_text(text: str) -> bool:
    """Checks if the text is very short (excluding known short phrases)."""
    if not text: return True
    known_short = ["Jesus wept.", "Rejoice always;", "Hallelujah.", "A.L.M.", "Ta-Ha.", "Ya Sin.", "Ha Mim.", "Ha-Mim."]
    text_check = text.lower().strip('. ')
    known_short_check = [ks.lower().strip('. ') for ks in known_short]
    if text_check in known_short_check: return False
    # Check for page header remnants like "SAMUEL 544..."
    if re.match(r'^[1-3]?\s?[A-Z\s.-]+?\s+\d+(\s*\.\.\.)?\s*$', text): return True
    # Check for other short junk like "(BCE)..."
    if re.match(r'^\(?[A-Z]{2,}\)?\s*\.\.\.$', text): return True

    return len(text.split()) < 3


# --- Main Script ---

def filter_jsonl(input_path: str, output_path: str):
    """Reads input JSONL, filters bad lines, writes to output."""
    lines_read = 0; lines_written = 0; lines_skipped = 0
    output_file_opened = False

    abs_input_path = os.path.abspath(input_path)
    abs_output_path = os.path.abspath(output_path)
    # print(f"DEBUG: Absolute Input Path: {abs_input_path}") # Keep commented unless needed
    # print(f"DEBUG: Absolute Output Path: {abs_output_path}")

    print(f"--- Starting Filter for: {input_path} ---")
    print(f"--- Attempting to write clean data to: {output_path} ---")

    try:
        # print(f"DEBUG: Attempting to open input '{input_path}' (read) and output '{output_path}' (write)...")
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:

            output_file_opened = True
            print(f"DEBUG: Successfully opened both files.")

            for i, line in enumerate(infile):
                lines_read += 1; line_num = i + 1; issues = []
                record = None
                try:
                    record = json.loads(line)
                    ref = record.get("ref", "")
                    text = record.get("text", "")

                    # --- Run Checks ---
                    if 'embedding' not in record or not isinstance(record.get('embedding'), list) or len(record.get('embedding', [])) != 1536:
                         issues.append(f"Missing/invalid embedding (len={len(record.get('embedding', []))})")
                    if 'id' not in record: issues.append("Missing ID")
                    if is_suspicious_reference(ref): issues.append(f"Suspicious reference: '{ref}'")
                    if is_short_text(text) and not is_suspicious_reference(ref): issues.append(f"Short text: '{text[:20]}...'")

                    # --- Write or Skip ---
                    if not issues:
                        try:
                            write(outfile, record) # Call the globally defined write function
                            lines_written += 1
                        except Exception as write_e:
                            print(f"FATAL ERROR: Failed to write line {line_num} to output file: {write_e}"); lines_skipped += 1
                    else:
                        lines_skipped += 1
                        # Reduce verbosity slightly for skipping
                        # print(f"L{line_num}: Skipping line - Issues: {'; '.join(issues)}")

                except json.JSONDecodeError:
                    lines_skipped += 1; print(f"L{line_num}: Skipping invalid JSON line.")
                except Exception as e:
                    lines_skipped += 1; rec_id = record.get('id', 'N/A') if record else 'N/A'
                    print(f"L{line_num}: Skipping due to error processing record (ID: {rec_id}): {e}")

    except FileNotFoundError: print(f"FATAL ERROR: Input file not found: {input_path}"); return
    except IOError as ioe: print(f"FATAL ERROR: Could not open/write file. Check permissions/path.\n Input: {input_path}\n Output: {output_path}\n Error: {ioe}"); return
    except Exception as e: print(f"FATAL ERROR during file operation: {e}"); return

    print("\n--- Filter Complete ---")
    print(f"Lines Read:    {lines_read}")
    print(f"Lines Written: {lines_written}")
    print(f"Lines Skipped: {lines_skipped}")
    if not output_file_opened and lines_read > 0: print(f"CRITICAL WARNING: Output file '{output_path}' might NOT have been created.")
    elif lines_written == 0 and lines_read > 0: print(f"CRITICAL WARNING: No lines written to '{output_path}'. All skipped.")
    elif lines_written > 0: print(f"Successfully wrote clean data to '{output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter a _embed.jsonl file based on quality checks.")
    parser.add_argument("input_file", help="Path to the input _embed.jsonl file.")
    parser.add_argument("output_file", help="Path for the output _clean_embed.jsonl file.")
    args = parser.parse_args()

    filter_jsonl(args.input_file, args.output_file)

