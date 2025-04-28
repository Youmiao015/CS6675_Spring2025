import requests
import time
import argparse

def get_subcategories(category, visited=None, path=None, max_depth=None, current_depth=0):
    """Recursively get all subcategories of a given category."""
    if visited is None:
        visited = set()
    
    if path is None:
        path = [category]
    
    # If we've already processed this category, return empty set to avoid cycles
    if category in visited:
        return set()
    
    # Check if we've reached the maximum recursion depth
    if max_depth is not None and current_depth >= max_depth:
        print(f"Reached max depth ({max_depth}) at: {category}")
        return set()
    
    # Add this category to visited set
    visited.add(category)
    
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": "subcat",
        "cmlimit": "500",
        "format": "json"
    }
    
    all_subcats = set()
    
    try:
        response = requests.get(url, params=params).json()
        subcategories = [item['title'] for item in response['query']['categorymembers']]
        
        # Add direct subcategories
        all_subcats.update(subcategories)
        
        # Process each subcategory recursively
        for subcat in subcategories:
            current_path = path + [subcat]
            path_str = " → ".join(current_path)
            print(f"Processing subcategory: {subcat}")
            print(f"Path: {path_str}")
            print(f"Depth: {current_depth + 1}")
            
            # Add a small delay to avoid overloading the API
            time.sleep(0.5)
            
            # Recursively get subcategories and add them to our set
            sub_subcats = get_subcategories(
                subcat, 
                visited, 
                current_path, 
                max_depth, 
                current_depth + 1
            )
            all_subcats.update(sub_subcats)
            
    except Exception as e:
        print(f"Error processing {category}: {e}")
    
    return all_subcats

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Recursively scrape Wikipedia categories')
    parser.add_argument('--max-depth', type=int, default=None, 
                        help='Maximum recursion depth (default: unlimited)')
    parser.add_argument('--root-category', type=str, 
                        default="Category:Artificial_intelligence",
                        help='Root category to start scraping from')
    parser.add_argument('--output', type=str, default="ai_subfields.txt",
                        help='Output file name')
    
    args = parser.parse_args()
    
    print(f"Starting recursive scrape from {args.root_category}")
    if args.max_depth:
        print(f"Maximum recursion depth: {args.max_depth}")
    else:
        print("No maximum recursion depth set (will scrape all levels)")
    
    all_categories = get_subcategories(args.root_category, max_depth=args.max_depth)
    
    # Add the root category to the result
    all_categories.add(args.root_category)
    
    # Sort categories for readability
    sorted_categories = sorted(all_categories)

    # Remove "Category:" prefix from each category
    sorted_categories = [category.replace("Category:", "") for category in sorted_categories]
    
    # Save to file
    with open(args.output, "w") as f:
        for category in sorted_categories:
            f.write(f"{category}\n")
    
    print(f"Scraping complete. Found {len(all_categories)} categories.")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
