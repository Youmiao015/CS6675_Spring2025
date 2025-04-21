#!/usr/bin/env python3
import pickle
import json

# Input and output file configuration
INPUT_META = 'papers_meta.pkl'
OUTPUT_JSONL = 'merged_all_fields.jsonl'  # You can rename it to merged_all_fields.jsonl for clarity

# Load all metadata records of papers from the pickle file
with open(INPUT_META, 'rb') as infile:
    papers_meta = pickle.load(infile)

print(f"Loaded {len(papers_meta)} records from {INPUT_META}")

# Write each record into the JSONL file, one record per line (ensure all fields are preserved)
with open(OUTPUT_JSONL, 'w', encoding='utf-8') as outfile:
    for record in papers_meta:
        json_line = json.dumps(record, ensure_ascii=False)
        outfile.write(json_line + "\n")

print(f"Successfully written all records to {OUTPUT_JSONL}")