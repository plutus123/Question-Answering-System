import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

# To avoid scraping the same URL multiple times, we'll use a set.
visited_links = set()
scraped_data = []

def scrape_data(url, depth):
    if depth > 5 or url in visited_links:
        return
    
    visited_links.add(url)
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the data from the current URL
        print(f"Scraping URL: {url}")
        page_content = soup.get_text(separator="\n")
        # scraped_data.append(f"URL: {url}\n{page_content[:500]}\n{'-'*80}\n") #(for testing) I am saving only 500 characters from each page since it is taking too much time to save entire content
        scraped_data.append(f"URL: {url}\n{page_content}\n{'-'*80}\n")
        # Find all sub-links on the current page
        for link in soup.find_all('a', href=True):
            sub_link = urljoin(url, link['href'])
            parsed_url = urlparse(sub_link)

            # Ensure the sub_link is within the same domain and is a valid URL
            if parsed_url.netloc == urlparse(url).netloc and parsed_url.scheme in ["http", "https"]:
                scrape_data(sub_link, depth + 1)
    
    except requests.RequestException as e:
        print(f"Failed to retrieve URL: {url} due to {e}")
    except Exception as e:
        print(f"An error occurred while scraping {url}: {e}")

# Start scraping from the parent URL
parent_url = "https://docs.nvidia.com/cuda/"
scrape_data(parent_url, 0)

# Save the parsed data into a text file
with open('scraped_data.txt', 'w', encoding='utf-8') as file:
    for data in scraped_data:
        file.write(data)

print('Scraping completed. Data saved to scraped_data.txt')
