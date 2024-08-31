import os
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import git
from .vectorsearch import VectorSearch 

import yaml

def import_url(url: str, vector_search: VectorSearch) -> None:
    config = load_config()
    local_directory = config['document_store']['local_directory']
    
    parsed_url = urlparse(url)
    
    if parsed_url.netloc in ['github.com', 'gitlab.com']:
        import_repository(url, local_directory, vector_search)
    elif url.lower().endswith('.pdf'):
        import_pdf(url, local_directory, vector_search)
    else:
        import_webpage(url, local_directory, vector_search)

def import_repository(url: str, local_directory: str, vector_search: VectorSearch) -> None:
    repo_name = os.path.basename(urlparse(url).path)
    local_repo_path = os.path.join(local_directory, repo_name)
    
    try:
        git.Repo.clone_from(url, local_repo_path)
        print(f"Cloned repository: {url}")
        
        for root, dirs, files in os.walk(local_repo_path):
            for file in files:
                if file.endswith(('.py', '.js', '.java', '.cpp', '.html', '.css', '.md', '.lua')):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    doc_id = os.path.relpath(file_path, local_repo_path)
                    metadata = {
                        'source': url,
                        'type': 'repository',
                        'file': doc_id
                    }
                    vector_search.add_document(doc_id, content, metadata)
        
        print(f"Imported repository contents: {url}")
    except git.GitCommandError as e:
        print(f"Failed to clone repository: {url}")
        print(f"Error: {e}")

def import_pdf(url: str, local_directory: str, vector_search: VectorSearch) -> None:
    response = requests.get(url)
    if response.status_code == 200:
        filename = os.path.join(local_directory, os.path.basename(url))
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        reader = PdfReader(filename)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        doc_id = os.path.basename(url)
        metadata = {
            'source': url,
            'type': 'pdf'
        }
        vector_search.add_document(doc_id, text, metadata)
        print(f"Imported PDF: {url}")
    else:
        print(f"Failed to download PDF: {url}")

def import_webpage(url: str, local_directory: str, vector_search: VectorSearch) -> None:
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        
        # Create a filename for the webpage
        filename = f"{urlparse(url).netloc.replace('.', '_')}.txt"
        file_path = os.path.join(local_directory, filename)
        
        # Save the webpage content to a file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        doc_id = urlparse(url).netloc
        metadata = {
            'source': url,
            'type': 'webpage',
            'file_path': file_path
        }
        vector_search.add_document(doc_id, text, metadata)
        print(f"Imported webpage: {url}")
    else:
        print(f"Failed to download webpage: {url}")

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        if not config:
            raise ValueError("Config file is empty")
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None
