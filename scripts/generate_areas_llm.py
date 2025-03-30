import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time
import argparse

def load_model():
    """Load the Qwen2.5-7B-Instruct model and tokenizer"""
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    # Load tokenizer separately
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir="/home/hice1/yliu3390/scratch/.cache/huggingface" # Keep cache dir if needed
        # trust_remote_code=True removed based on snippet
    )
    # Load model with torch_dtype and trust_remote_code
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto", # Added based on snippet
        device_map="auto",
        trust_remote_code=True, # Keep for Qwen model code
        cache_dir="/home/hice1/yliu3390/scratch/.cache/huggingface", # Keep cache dir if needed
        attn_implementation="flash_attention_2"
    ).eval() # Keep eval mode for inference
    return model, tokenizer

def generate_research_areas(model, tokenizer, words, num_areas=3):
    """Generate research areas for a list of words using the model"""
    # Create example area strings dynamically for the prompt example
    example_areas = ["\"Federated Learning\"", "\"Retrieval Augmented Generation\"", "\"Diffusion Models\""]
    example_areas_str = ',\n        '.join(example_areas)

    # Construct the user prompt content with updated instructions
    user_prompt = f"""Given the following list of named entities, please identify specific computer science research areas (subfields or subsubfields) that best represent these entities.
Generate *up to* {num_areas} distinct research areas. If fewer areas accurately summarize the core topic, return fewer.
If the entities clearly *do not* represent computer science research areas, return an empty list. 
Avoid generating areas that are too broad or general like 'Computer Science', 'Artificial Intelligence', or 'Computer Vision'.
Format your response *only* as a JSON object with a single key "research_areas" containing an array of strings. Do not include any introductory text or explanation outside the JSON structure.

Entities: {', '.join(words)}

Example of a valid response format (for up to {num_areas} areas):
{{
    "research_areas": [
        {example_areas_str}
    ]
}}

Example of a response if entities are not CS-related:
{{
    "research_areas": []
}}
"""

    # Prepare messages for chat template - updated system prompt
    messages = [
        {"role": "system", "content": f"You are a helpful assistant skilled in identifying computer science research areas. You must generate responses *only* in the requested JSON format. Return up to {num_areas} areas, or an empty list if the input is not related to CS research. Avoid generating areas that are too broad or general like 'Computer Science', 'Artificial Intelligence', or 'Computer Vision'."},
        {"role": "user", "content": user_prompt}
    ]

    # Apply chat template and tokenize
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate response using model.generate
    # Set max_new_tokens reasonably high enough for the JSON output
    # Pad token id is often needed for batch generation, setting it here
    # EOS token id helps the model know when to stop
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256, # Increased slightly for safety margin
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, # Handle potential missing pad_token
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode the generated tokens, skipping the input tokens
    input_token_len = model_inputs.input_ids.shape[1]
    generated_ids = generated_ids[:, input_token_len:]

    # Decode the response
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Updated JSON parsing logic
    try:
        # Extract JSON from response
        response_trimmed = response.strip()
        start_idx = response_trimmed.find('{')
        end_idx = response_trimmed.rfind('}') + 1
        if start_idx == -1 or end_idx == 0:
             raise ValueError("JSON object not found in response")
        json_str = response_trimmed[start_idx:end_idx]
        result = json.loads(json_str)

        # Validation: Check if 'research_areas' key exists and is a list
        if "research_areas" in result and isinstance(result["research_areas"], list):
             # Check if all elements in the list are strings
             if all(isinstance(area, str) for area in result["research_areas"]):
                 # Accept the list as is (can be empty or up to num_areas)
                 # Optional: Limit the length just in case the model returns more than requested
                 return result["research_areas"][:num_areas]
             else:
                 print(f"Error: 'research_areas' list contains non-string elements.")
                 print(f"Raw response: {response}")
                 print(f"Parsed JSON: {result}")
                 # Return error list with expected max length for consistency downstream? Or an empty list?
                 # Let's return a specific error list matching num_areas.
                 return ["Error: Non-string element in list"] * num_areas
        else:
             print(f"Error: Response JSON does not contain a valid 'research_areas' list.")
             print(f"Raw response: {response}")
             print(f"Parsed JSON: {result}")
             return ["Error: Invalid format or missing key"] * num_areas

    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Raw response: {response}")
        # Ensure the error list matches the expected max length
        return [f"Error parsing response: {e}"] * num_areas

def main():
    # --- Argument Parsing Start ---
    parser = argparse.ArgumentParser(description="Process word clusters to generate research areas using an LLM.")
    parser.add_argument(
        "--max_clusters",
        type=int,
        default=None,
        help="Maximum number of clusters to process (default: process all)."
    )
    parser.add_argument(
        "--num_areas",
        type=int,
        default=3,
        help="Number of research areas to generate per cluster (default: 3)."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/step5_hdbscan_umap_clusters.json",
        help="Path to the input JSON file containing clusters."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/cluster_dict_with_llm_generated_areas.json",
        help="Path to the output JSON file."
    )
    args = parser.parse_args()
    # --- Argument Parsing End ---

    # Load the model
    print("Loading model...")
    model, tokenizer = load_model()

    # Load the clusters
    print(f"Loading clusters from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        clusters = json.load(f)

    # --- Load existing results if output file exists ---
    results = {}
    try:
        with open(args.output_file, 'r') as f:
            results = json.load(f)
            print(f"Loaded {len(results)} existing results from {args.output_file}")
    except FileNotFoundError:
        print(f"Output file {args.output_file} not found, starting fresh.")
    except json.JSONDecodeError:
        print(f"Error reading {args.output_file}, starting fresh.")


    # Process each cluster
    processed_count = 0
    clusters_to_process = list(clusters.items()) # Convert to list to allow slicing

    # Determine which clusters need processing
    items_to_process = [
        (cid, words) for cid, words in clusters_to_process
        if cid not in results or len(results[cid].get("research_areas", [])) != args.num_areas
    ]

    # Apply max_clusters limit if set
    if args.max_clusters is not None:
        print(f"Limiting processing to a maximum of {args.max_clusters} new clusters.")
        items_to_process = items_to_process[:args.max_clusters]

    total_to_process = len(items_to_process)
    print(f"Total clusters to process: {total_to_process}")

    for cluster_id, words in tqdm(items_to_process, desc="Processing clusters", total=total_to_process):
        print(f"\nProcessing cluster {cluster_id} with {len(words)} words")

        # Generate research areas
        research_areas = generate_research_areas(model, tokenizer, words, args.num_areas)

        # Store results
        results[cluster_id] = {
            "words": words, # Store words for context, maybe limit length if too long?
            "research_areas": research_areas
        }

        # Save intermediate results after each cluster
        try:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"\nError saving intermediate results: {e}")

        processed_count += 1

        # Add a small delay to avoid potential rate limiting or excessive GPU heat
        time.sleep(1)

    print(f"\nProcessed {processed_count} new clusters.")
    print(f"Processing complete! Results saved to {args.output_file}")

if __name__ == "__main__":
    main() 