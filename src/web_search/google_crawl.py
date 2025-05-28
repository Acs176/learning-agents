from googlesearch import search
import requests
from bs4 import BeautifulSoup

def extract_title_and_text(html):
    """
    Given raw HTML, returns a tuple (title, text_content)
    where `title` is the page title (or None),
    and `text_content` is all the <p> text joined by newlines.
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # 1. Extract <title>
    title_tag = soup.title
    title = title_tag.string.strip() if title_tag and title_tag.string else None
    
    # 2. Extract all <p> tags
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    text_content = "\n\n".join(paragraphs).strip()
    
    return title, text_content
def get_top_results(query, n):
    urls = search(query, num_results=n)
    results = []
    for url in urls:
        try:
            result = requests.get(url)
            if result.status_code >= 300:
                print(f"Error fetching url [{result.text}]")
                continue
            title, text = extract_title_and_text(result.text)
            results.append("\n".join([title, text]))
        except Exception as e:
            print(f"Exception raised: {e}")
    
    return results