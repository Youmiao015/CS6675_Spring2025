#!/usr/bin/env python3
import json
from tqdm import tqdm

# Path to the original data file (make sure the data is downloaded to this directory)
input_file = "arxiv_dataset/arxiv-metadata-oai-snapshot.json"
output_file = "cs_papers.json"
filtered_papers = []

print("Reading dataset and filtering CS papers...")
with open(input_file, "r", encoding="utf-8") as file:
    for line in tqdm(file, desc="Reading dataset"):
        data = json.loads(line)
        # Filter papers with category starting with "cs"
        if data.get('categories', '')[:2] != 'cs':
            continue
        # Ensure both title and abstract are present
        if 'title' not in data or 'abstract' not in data:
            continue
        filtered_papers.append(data)

print(f"Total CS papers: {len(filtered_papers)}")

# Save the filtered results in JSON Lines format
with open(output_file, "w", encoding="utf-8") as out_f:
    for paper in filtered_papers:
        json_line = json.dumps(paper, ensure_ascii=False)
        out_f.write(json_line + "\n")

print(f"Filtered data saved to {output_file}")