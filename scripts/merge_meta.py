#!/usr/bin/env python3
import os
import glob
import pickle
import json
import re

# Filename pattern to match partial files
PART_PATTERN = "papers_meta_part_*.pkl"
# Output file containing only title and abstract fields
OUTPUT_FILE = "merged_title_abstract.jsonl"

def extract_part_number(filename):
    """Extract the numeric part from the filename, e.g. papers_meta_part_15.pkl -> 15"""
    match = re.search(r"papers_meta_part_(\d+)\.pkl", filename)
    if match:
        return int(match.group(1))
    return -1

def main():
    # Find and sort all matching part files
    part_files = glob.glob(PART_PATTERN)
    if not part_files:
        print(f"No files found matching pattern: {PART_PATTERN}")
        return
    
    part_files_sorted = sorted(part_files, key=extract_part_number)
    print("Found and sorted part files:")
    for f in part_files_sorted:
        print(f)
    
    # For each part file, keep only title and abstract fields and write to the output file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for pf in part_files_sorted:
            print(f"Merging {pf} ...")
            with open(pf, 'rb') as f:
                records = pickle.load(f)
            
            for record in records:
                filtered_record = {
                    "title": record.get("title", ""),
                    "abstract": record.get("abstract", "")
                }
                out_f.write(json.dumps(filtered_record, ensure_ascii=False))
                out_f.write('\n')
    
    print(f"All part files merged into {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()