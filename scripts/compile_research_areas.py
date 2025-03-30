import json
import csv
from collections import Counter
import argparse
import os

def analyze_areas(input_json_path, output_csv_path):
    """
    Parses the research areas JSON, counts frequencies, and writes to a CSV file.

    Args:
        input_json_path (str): Path to the input JSON file (cluster_research_areas.json).
        output_csv_path (str): Path to the output CSV file.
    """
    print(f"Reading clusters from: {input_json_path}")
    try:
        with open(input_json_path, 'r') as f:
            clusters_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_json_path}")
        return

    area_counter = Counter()
    total_clusters_processed = 0
    clusters_with_areas = 0

    print("Counting research area frequencies...")
    for cluster_id, data in clusters_data.items():
        total_clusters_processed += 1
        if "research_areas" in data and isinstance(data["research_areas"], list):
            has_valid_area = False
            for area in data["research_areas"]:
                # Check if the area is a valid string and not an error message
                if isinstance(area, str) and not area.lower().startswith("error:"):
                    # Normalize whitespace and count
                    normalized_area = ' '.join(area.strip().split())
                    if normalized_area: # Ensure it's not empty after stripping
                        area_counter[normalized_area] += 1
                        has_valid_area = True
            if has_valid_area:
                clusters_with_areas += 1
        else:
            print(f"Warning: Cluster {cluster_id} missing 'research_areas' list or has invalid format.")

    print(f"Processed {total_clusters_processed} clusters.")
    print(f"{clusters_with_areas} clusters contained valid research areas.")
    print(f"Found {len(area_counter)} unique research areas.")

    if not area_counter:
        print("No valid research areas found to write to CSV.")
        return

    # Sort areas by frequency in descending order
    sorted_areas = sorted(area_counter.items(), key=lambda x: x[1], reverse=True)

    print(f"Writing frequencies to: {output_csv_path}")
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['Research Area', 'Frequency'])
            # Write data
            writer.writerows(sorted_areas)
        print("CSV file written successfully.")
    except IOError as e:
        print(f"Error writing CSV file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze research area frequencies from cluster JSON.")
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/cluster_dict_with_llm_generated_areas.json",
        help="Path to the input JSON file containing cluster research areas."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/llm_generated_areas.csv",
        help="Path to the output CSV file."
    )
    args = parser.parse_args()

    analyze_areas(args.input_file, args.output_file) 