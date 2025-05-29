import os
import requests
from serpapi import Client
from bs4 import BeautifulSoup

def extract_title_and_text(html):
    """
    Given raw HTML, returns a tuple (title, text_content)
    where `title` is the page title (or None),
    and `text_content` is all the <p> text joined by newlines.
    """
    soup = BeautifulSoup(html, "html.parser")
    
    title_tag = soup.title
    title = title_tag.string.strip() if title_tag and title_tag.string else None
    
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    text_content = "\n\n".join(paragraphs).strip()
    
    return title, text_content

def get_top_results(query, n, api_key=None):
    """
    Uses SerpAPI Client to get the top n organic Google results for `query`,
    then fetches and parses each page into a single string:
      "<title>\\n<text_content>"
    """
    # 1. Retrieve API key
    if api_key is None:
        api_key = os.environ.get("SERPAPI_API_KEY")
        if not api_key:
            raise ValueError("Set SERPAPI_API_KEY env var or pass api_key explicitly.")
    
    # 2. Initialize SerpAPI client
    client = Client()  # no args here
    
    # 3. Perform the search (include api_key in params)
    params = {
        "engine":      "google",
        "q":           query,
        "num":         n,
        "api_key":     api_key,
    }
    response = client.search(params)
    organic = response.get("organic_results", [])
    
    # 4. Fetch and parse each URL
    results = []
    for entry in organic[:n]:
        url = entry.get("link")
        if not url:
            continue
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            title, text = extract_title_and_text(resp.text)
            combined = "\n".join([t for t in (title, text) if t])
            results.append(combined)
        except requests.RequestException as e:
            print(f"Error fetching {url!r}: {e}")
        except Exception as e:
            print(f"Error parsing {url!r}: {e}")
    
    return results
