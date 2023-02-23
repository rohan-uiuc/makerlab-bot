import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import config.config
from .text_utils import clean_text, encode_text
from ftfy import fix_text
import os
def get_links(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')

    links = soup.find_all('a')
    paths = [link.get('href') for link in links]

    paths = [path for path in paths if path is not None and path.startswith('/')]

    return paths

def get_contents(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')

    paragraphs = soup.find_all('p')
    text = '\n'.join([p.get_text() for p in paragraphs])

    # Fine-tune the model on the text data
    index = url.split('/')[-1]

    #encoded_text = encode_text(text)
    text = fix_text(text)
    path = config.config.DATA_MODEL_DIR
    filename = os.path.join(path, f'{index}.txt')
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)

    f.close()

    links = get_links(url)
    base_url = url.rsplit('/', 1)[0]

    for link in links:
        full_link = urljoin(base_url, link)
        get_contents(full_link)

    return fix_text(text)
