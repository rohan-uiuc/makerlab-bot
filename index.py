import requests, os
from urllib.parse import urlparse, urljoin
from gpt_index import GPTSimpleVectorIndex, BeautifulSoupWebReader, Document
from bs4 import BeautifulSoup

# @sleep_and_retry
# @limits(calls=6, period=60)
# def process_batch(documents):
#
#     return index
index_file = './doc_qa.json'
def create_index():
    if not os.path.isfile(index_file):
        #@markdown ### Enter the url to your documentation:
        Input_URL = "https://makerlab.illinois.edu" #@param {type:"string"}

        # Extract all links from the page using BeautifulSoup
        parsed_url = urlparse(Input_URL)
        base_url = parsed_url.scheme + "://" + parsed_url.netloc
        response = requests.get(Input_URL)
        soup = BeautifulSoup(response.content, 'html.parser')
        paths = set([a.get('href') for a in soup.find_all('a', href=True)])

        # Filter out invalid links and join them with the base URL
        links = []
        for path in paths:
            url = urljoin(base_url, path)
            parsed_url = urlparse(url)
            if parsed_url.scheme in ["http", "https"] and "squarespace" not in parsed_url.netloc:
                links.append(url)

        print("Links extracted")
        # Read the contents of each link into a list of documents
        documents = BeautifulSoupWebReader().load_data(links)

        # Combine the text of all documents into a single string
        all_text = ' '.join(doc.text for doc in documents)

        # Create a single Document object from the combined text
        combined_doc = Document(all_text)

        # Create and save the GPT index
        index = GPTSimpleVectorIndex([combined_doc])

        # Save the GPT index to disk
        index.save_to_disk(index_file)

    # Load the GPT index from disk
    index2 = GPTSimpleVectorIndex.load_from_disk(index_file)
    return index2
