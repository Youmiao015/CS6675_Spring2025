import json
import requests
import fitz  # PyMuPDF
import re
import os
import time
from pathlib import Path
import tempfile
import sys

def count_papers():
    """Count total number of papers in the metadata file."""
    count = 0
    with open('../.cache/kagglehub/datasets/Cornell-University/arxiv/versions/226/arxiv-metadata-oai-snapshot.json', 'r') as f:
        for line in f:
            paper_data = json.loads(line)
            category = paper_data.get('categories', '').lower()
            if category.startswith('cs'):
                count += 1
    return count

def get_metadata():
    with open('../.cache/kagglehub/datasets/Cornell-University/arxiv/versions/226/arxiv-metadata-oai-snapshot.json', 'r') as f:
        for line in f:
            yield line

def load_processed_papers(processed_file):
    """Load the set of already processed paper IDs from a file."""
    processed_papers = set()
    if os.path.exists(processed_file):
        with open(processed_file, 'r') as f:
            for line in f:
                paper_id = line.strip()
                if paper_id:
                    processed_papers.add(paper_id)
    return processed_papers

def save_processed_paper(processed_file, paper_id):
    """Save a processed paper ID to the file."""
    with open(processed_file, 'a') as f:
        f.write(f"{paper_id}\n")

def download_pdf(arxiv_id):
    """Download PDF from arXiv to a local file in the data/temp_pdfs directory."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    
    try:
        response = requests.get(pdf_url)
        if response.status_code == 200:
            # Create a local directory for PDFs if it doesn't exist
            local_dir = Path("data/temp_pdfs")
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / f"{arxiv_id}.pdf"
            
            # Write the content to the file
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            print(f"Downloaded PDF for {arxiv_id} to {local_path}")
            return str(local_path)
    except Exception as e:
        print(f"Error downloading {arxiv_id}: {e}")
    return None

def find_discussion_section(pdf_path):
    """
    Attempts to find and extract the Discussion/Future Work section from a PDF.
    Returns tuple of (discussion_text, detected_keyword)
    """
    print(f"Finding discussion section for {pdf_path}")
    if os.path.exists(pdf_path):
        print(f"PDF path exists: {pdf_path}")
    else:
        print(f"PDF path does not exist: {pdf_path}")
    doc = fitz.open(pdf_path)
    print(f"Opened document for {pdf_path}")
    discussion_text = ""
    found_discussion_header = False
    detected_keyword = None
    possible_next_headers = ["conclusion", "summary", "acknowledgement", "acknowledgment", "references", "bibliography"]
    potential_header_keywords = ["future work", "future direction", "future research", "discussion"]
    word_count = 0
    max_words = 1000  # Limit to 1000 words

    # Simple heuristic: try to guess body font size on first few pages
    body_font_sizes = {}
    for page_num in range(min(5, len(doc))):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    if "spans" in line:
                        for span in line["spans"]:
                            size = round(span["size"])
                            body_font_sizes[size] = body_font_sizes.get(size, 0) + len(span["text"])
    
    body_font_size = max(body_font_sizes.items(), key=lambda x: x[1])[0] if body_font_sizes else None

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
        
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = "".join(span["text"] for span in line["spans"]).strip()
                    line_text_lower = line_text.lower()
                    
                    # Header Identification Logic
                    is_potential_header = False
                    if re.match(r"^\d+\.?\s+", line_text) or re.match(r"^[IVXLCDM]+\.?\s+", line_text):
                        is_potential_header = True
                    elif body_font_size and "spans" in line and line["spans"]:
                        span_size = round(line["spans"][0]["size"])
                        is_bold = 'bold' in line["spans"][0]["font"].lower()
                        if span_size > body_font_size * 1.05 or is_bold:
                            if len(line_text.split()) < 8:
                                is_potential_header = True
                    elif line_text.isupper() and len(line_text.split()) < 8 and len(line_text) > 3:
                        is_potential_header = True

                    # Section Start/End Logic
                    if is_potential_header:
                        if found_discussion_header:
                            header_tokens = re.findall(r'\b\w+\b', line_text_lower)
                            if any(token in header_tokens for token in possible_next_headers):
                            # if any(token in possible_next_headers for token in header_tokens):
                                doc.close()
                                return discussion_text.strip(), detected_keyword
                        
                        # Check if this header marks the start of the discussion
                        for keyword in potential_header_keywords:
                            if keyword in line_text_lower:
                                found_discussion_header = True
                                detected_keyword = keyword
                                break
                        if found_discussion_header:
                            continue
                            
                    if found_discussion_header:
                        # Count words in the current line
                        words_in_line = len(line_text.split())
                        
                        # Check if adding this line would exceed the word limit
                        if word_count + words_in_line > max_words:
                            # Add only enough words to reach the limit
                            remaining_words = max_words - word_count
                            if remaining_words > 0:
                                words = line_text.split()[:remaining_words]
                                discussion_text += " ".join(words) + "\n"
                                word_count = max_words
                            break
                        else:
                            discussion_text += line_text + "\n"
                            word_count += words_in_line
                            
                        # If we've reached the word limit, stop processing
                        if word_count >= max_words:
                            break

    doc.close()
    return discussion_text.strip(), detected_keyword

def main():
    # Create output directory for discussions
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open JSON file for writing discussions
    discussions_file = output_dir / "discussions.json"
    
    # Define file for tracking processed papers
    processed_file = output_dir / "processed_papers.txt"
    
    # Load already processed papers
    processed_papers = load_processed_papers(processed_file)
    print(f"Found {len(processed_papers)} already processed papers")
    
    # Count total papers first
    # total_papers = count_papers()
    # print(f"Found {total_papers} CS papers to process")
    
    metadata = get_metadata()
    success_count = 0
    processed_count = 0
    
    for paper in metadata:
        paper_data = json.loads(paper)
        
        category = paper_data.get('categories', '').lower()
        if not category.startswith("cs"):
            continue
            
        arxiv_id = paper_data.get('id')
        
        if not arxiv_id:
            continue
            
        # Skip if already processed
        if arxiv_id in processed_papers:
            print(f"Skipping already processed paper: {arxiv_id}")
            continue
            
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} CS papers so far")
        
        # Download PDF to temporary file
        pdf_path = download_pdf(arxiv_id)
        if not pdf_path:
            continue
            
        # Extract discussion section
        try:
            discussion, keyword = find_discussion_section(pdf_path)
            if discussion:
                # Append to JSON file
                with open(discussions_file, 'a', encoding='utf-8') as f:
                    json.dump({
                        "id": arxiv_id,
                        "discussion": discussion,
                        "keyword": keyword
                    }, f)
                    f.write('\n')
                print(f"Successfully extracted discussion for {arxiv_id} using keyword: {keyword}")
                success_count += 1
                
                # Save to processed papers file
                save_processed_paper(processed_file, arxiv_id)
            else:
                print(f"No discussion section found for {arxiv_id}")
                # Still mark as processed even if no discussion found
                save_processed_paper(processed_file, arxiv_id)
        except Exception as e:
            print(f"Error processing {arxiv_id}: {e}")
            # Exit the program when an exception occurs
            sys.exit(1)
        finally:
            # Clean up temporary PDF file
            try:
                os.unlink(pdf_path)
            except:
                pass
            
        # Rate limiting to be nice to arXiv
        time.sleep(3)
        
    print(f"Finished processing {processed_count} CS papers")
    print(f"Successfully extracted {success_count} discussions")

if __name__ == "__main__":
    main()