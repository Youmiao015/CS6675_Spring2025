import requests

def download_arxiv_pdf(arxiv_id, filename=None):
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    if filename is None:
        filename = f"{arxiv_id.replace('/', '_')}.pdf"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download: {arxiv_id} (status code: {response.status_code})")

# Example usage
download_arxiv_pdf("0705.0599")
