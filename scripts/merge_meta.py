#!/usr/bin/env python3
import os
import glob
import pickle
import json
import re

# Directory containing the part files (if needed)
# For simplicity, assume we're in the same directory as the part files.
PART_PATTERN = "papers_meta_part_*.pkl"
OUTPUT_FILE = "merged.jsonl"

def extract_part_number(filename):
    """Extract the numeric part from the filename, e.g. papers_meta_part_15.pkl -> 15."""
    match = re.search(r"papers_meta_part_(\d+)\.pkl", filename)
    if match:
        return int(match.group(1))
    return -1

def main():
    # Find and sort part files by their numeric index
    part_files = glob.glob(PART_PATTERN)
    if not part_files:
        print(f"No files found matching pattern: {PART_PATTERN}")
        return
    
    part_files_sorted = sorted(part_files, key=extract_part_number)
    print("Found and sorted part files:")
    for f in part_files_sorted:
        print(f)
    
    # Open the final merged JSONL file in write mode
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for pf in part_files_sorted:
            print(f"Merging {pf} ...")
            # Load one batch from the .pkl file
            with open(pf, 'rb') as f:
                records = pickle.load(f)
            
            # Write each record as a JSON line to merged.jsonl
            for record in records:
                out_f.write(json.dumps(record, ensure_ascii=False))
                out_f.write('\n')
            
            # At this point, we've finished one batch and freed its memory
    
    print(f"All part files merged into {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()