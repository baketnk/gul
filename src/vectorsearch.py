import json
import os
from typing import List, Dict
from .embed_base import BaseEmbeddingModel
import re
from tqdm import tqdm
from collections import deque
import numpy as np

class VectorSearch:
    def __init__(self, embedding_model: BaseEmbeddingModel, embedding_file: str = 'embeddings.json', max_length: int = 512):
        self.embedding_model = embedding_model
        self.embedding_file = embedding_file
        self.embeddings = self.load_embeddings()
        self.tokenizer = embedding_model.tokenizer
        self.max_length = max_length

    def load_embeddings(self) -> Dict:
        if os.path.exists(self.embedding_file):
            with open(self.embedding_file, 'r') as f:
                return json.load(f)
        return {}

    
    def regenerate_embeddings(self, documents_dir: str):
        print("Regenerating embeddings for documents...")
        file_queue = deque()

        # First, build the queue of all files
        for root, _, files in os.walk(documents_dir):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, documents_dir)
                if "/.git" in relative_path:
                    continue
                file_queue.append((file_path, relative_path))

        # Now process the queue with a single tqdm progress bar
        total_files = len(file_queue)
        with tqdm(total=total_files, desc="Processing files") as pbar:
            while file_queue:
                file_path, relative_path = file_queue.popleft()
                
                if relative_path not in self.embeddings:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Chunk the content
                        chunks = self.embedding_model.chunk_text(content, self.max_length)
                        
                        for i, chunk in enumerate(chunks):
                            chunk_id = f"{relative_path}__chunk_{i}"
                            embedding = self.embed_text(chunk)
                            
                            metadata = {
                                'filename': relative_path,
                                'chunk_index': i,
                                'tags': self.get_tags(file_path, content),
                                'chunk': chunk,  
                            }
                            
                            self.embeddings[chunk_id] = {
                                'embedding': embedding,
                                'metadata': metadata
                            }
                    except UnicodeDecodeError:
                        print(f"Skipping file due to decoding error: {file_path}")
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
                
                pbar.update(1)
        
        self.save_embeddings()
        print("Embeddings regeneration completed.")

    def get_tags(self, file_path: str, content: str) -> List[str]:
        tags = []
        
        # File type
        _, file_extension = os.path.splitext(file_path)
        tags.append(f"file_type:{file_extension[1:]}")
        
        # Code language
        code_language = self.detect_code_language(file_path, content)
        if code_language:
            tags.append(f"language:{code_language}")
        
        # Document type
        doc_type = self.detect_document_type(file_path, content)
        tags.append(f"doc_type:{doc_type}")
        
        return tags

    def detect_code_language(self, file_path: str, content: str) -> str:
        file_extension = os.path.splitext(file_path)[1].lower()
        extension_to_language = {
            '.py': 'python',
            '.js': 'javascript',
            '.java': 'java',
            '.cpp': 'c++',
            '.c': 'c',
            '.html': 'html',
            '.css': 'css',
            '.lua': 'lua',
            '.rb': 'ruby',
            '.php': 'php',
            '.go': 'go',
            '.rs': 'rust',
            '.ts': 'typescript',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
            '.m': 'matlab',
            '.sql': 'sql'
        }
        return extension_to_language.get(file_extension, 'unknown')

    def detect_document_type(self, file_path: str, content: str) -> str:
        file_name = os.path.basename(file_path).lower()
        
        if file_name in ['readme.md', 'readme.txt']:
            return 'readme'
        elif 'license' in file_name:
            return 'license'
        elif file_name.endswith(('.md', '.txt', '.rst')):
            return 'documentation'
        elif file_name.endswith(('.py', '.js', '.java', '.cpp', '.c', '.lua', '.rb', '.go', '.rs', '.swift', '.kt', '.scala')):
            return 'source_code'
        elif file_name.endswith(('.html', '.css')):
            return 'web_content'
        elif file_name.endswith('.json'):
            return 'data_file'
        elif file_name.endswith(('.yml', '.yaml')):
            return 'configuration'
        else:
            return 'other'

    def save_embeddings(self):
        with open(self.embedding_file, 'w') as f:
            json.dump(self.embeddings, f, indent=2)

    def embed_text(self, text: str) -> List[float]:
        return self.embedding_model.embed_text(text)

    def add_document(self, doc_id: str, text: str, metadata: Dict):
        chunks = self.chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}__chunk_{i}"
            embedding = self.embed_text(chunk)
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = i
            chunk_metadata['chunk'] = chunk
            self.embeddings[chunk_id] = {
                'embedding': embedding,
                'metadata': chunk_metadata
            }
        self.save_embeddings()

    def cosine_sim(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = np.array(self.embed_text(query))
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize query embedding
 
        # Convert all document embeddings to a numpy array
        doc_embeddings = np.array([data['embedding'] for data in self.embeddings.values()])
        doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1)[:, np.newaxis]

        
        # Compute dot products in a single operation
        scores = np.dot(doc_embeddings, query_embedding)
        
        # Get top_k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        doc_ids = list(self.embeddings.keys())
        for idx in top_indices:
            doc_id = doc_ids[idx]
            metadata = self.embeddings[doc_id]['metadata']
            file_path = metadata.get('filename', '')
            
            # Read a snippet of the content
            snippet = metadata['chunk'] 
            
            results.append({
                'doc_id': doc_id,
                'score': float(scores[idx]),
                'metadata': metadata,
                'snippet': snippet
            })
        
        return results



