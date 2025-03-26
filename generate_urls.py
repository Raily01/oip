from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
from langdetect import detect


def is_russian(text):
    try:
        return detect(text) == "ru"
    except:
        return False


def get_page_soup(url):
    try:
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        }
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except requests.RequestException as e:
        print(f"Error {url}: {e}")
        return None


def get_links(soup, base_url):
    links = set()
    for link in soup.find_all('a', href=True):
        full_url = urljoin(base_url, link['href'])
        links.add(full_url)
    return links


def get_unique_urls(start_url, max_urls=150):
    visited_urls = set()
    urls_to_visit = [start_url]
    count_url = 0
    while urls_to_visit and len(visited_urls) < max_urls:
        current_url = urls_to_visit.pop(count_url)
        count_url += 1
        print(f"Visit URL №{count_url} - {current_url}")
        soup = get_page_soup(current_url)
        if not soup:
            continue
        text = soup.get_text().strip()
        if text and is_russian(text):
            visited_urls.add(current_url)
            links = get_links(soup, current_url)
            for link in links:
                if link not in visited_urls and link not in urls_to_visit and "#" not in link:
                    urls_to_visit.append(link)

                if len(visited_urls) >= max_urls:
                    break
            print(f"Total URLs of passed conditions: {len(visited_urls)}")

    return list(visited_urls)[:max_urls]


start_url = "https://ru.wikipedia.org/wiki/Сёрфинг"
unique_urls = get_unique_urls(start_url, 150)

with open('urls.txt', 'w') as f:
    for url in unique_urls:
        f.write("%s\n" % url)