# check_data.py
import json
import re
import sys
from pathlib import Path

def check_data_quality(file_path: Path):
    """
    Streams through a JSONL file and checks for data quality issues.
    """
    if not file_path.exists():
        print(f"ERROR: File not found at {file_path}")
        return

    print(f"--- Starting data quality check for: {file_path.name} ---")
    issue_count = 0
    
    # Regex to find verse references with verse numbers > 200 or chapter numbers > 200
    # This is a simple heuristic to find outliers like "10:494".
    ref_pattern = re.compile(r'(\d+):(\d+)')

    with file_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line_num = i + 1
            try:
                data = json.loads(line)
                
                # Check 1: Schema validation
                required_keys = ['ref', 'text', 'embedding']
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    print(f"L{line_num}: Missing keys - {', '.join(missing_keys)}")
                    issue_count += 1
                    continue

                # Check 2: Text quality
                text = data.get('text', '').strip()
                if not text or len(text.split()) < 3:
                    print(f"L{line_num}: Short or empty text - '{text[:50]}...'")
                    issue_count += 1

                # Check 3: Reference sanity check
                ref = data.get('ref', '')
                match = ref_pattern.search(ref)
                if match:
                    chapter, verse = int(match.group(1)), int(match.group(2))
                    if chapter > 200 or verse > 200:
                        print(f"L{line_num}: Suspicious reference number - '{ref}'")
                        issue_count += 1

            except json.JSONDecodeError:
                print(f"L{line_num}: Invalid JSON")
                issue_count += 1
    
    print(f"--- Check complete. Found {issue_count} potential issues. ---")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_data.py <path_to_jsonl_file>")
    else:
        check_data_quality(Path(sys.argv[1]))