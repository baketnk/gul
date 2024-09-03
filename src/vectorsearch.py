import json
import os
from typing import List, Dict
import numpy as np
from torch.backends.mps import is_available
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from collections import deque
import logging

BANNED_EXTENSIONS = [
".cff",
".cpp",
".csv",
".cu",
".cuh",
".dockerfile",
".h",
".idx",
".ipynb",
".json",
".jsonnet",
".len",
".lua",
".model",
".pack",
".png",
".py",
".pyx",
".rev",
".rockspec",
".sample",
".sh",
".source",
".target",
".template",
".toml",
".tsv",
".vim",
".yaml",
".yml",
]

class VectorSearch:
    def __init__(self, model_name: str = 'mixedbread-ai/mxbai-embed-large-v1', embedding_file: str = 'embeddings.json', max_length: int = 512):
        self.model_name = model_name
        self.embedding_file = embedding_file
        self.max_length = max_length
        self.embeddings = self.load_embeddings()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model.to(self.device)

    def load_embeddings(self) -> Dict:
        if os.path.exists(self.embedding_file):
            with open(self.embedding_file, 'r') as f:
                return json.load(f)
        return {}

    def save_embeddings(self):
        with open(self.embedding_file, 'w') as f:
            json.dump(self.embeddings, f, indent=2)

    def embed_text(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_length, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0].tolist()
        return embedding

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = np.array(self.embed_text(f"Represent this sentence for searching relevant passages: {query}"))
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        doc_embeddings = np.array([data['embedding'] for data in self.embeddings.values()])
        doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1)[:, np.newaxis]

        # Calculate cosine similarity
        cosine_similarities = np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        top_indices = np.argsort(cosine_similarities)[-top_k:][::-1]

        results = []
        doc_ids = list(self.embeddings.keys())
        for idx in top_indices:
            chunk_id = doc_ids[idx]
            metadata = self.embeddings[chunk_id]['metadata']
            doc_id = chunk_id.rsplit('_chunk_', 1)[0]
            results.append({
                'doc_id': doc_id,
                'chunk_id': chunk_id,
                'score': float(cosine_similarities[idx]),
                'metadata': metadata
            })

        return results
    def chunk_text(self, text: str):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        encoded_chunks = []
        chunk_size = 512
        overlap = 128
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk = tokens[i:i + chunk_size]
            chunks.append(self.tokenizer.decode(chunk))
            encoded_chunks.append(chunk)
        
        return chunks, encoded_chunks

    def add_document(self, doc_id: str, text: str, metadata: Dict):
        chunks, _ = self.chunk_text(text)
        embeddings = []
        for i, chunk in enumerate(chunks):
            embedding = self.embed_text(chunk)
            embeddings.append(embedding)
            chunk_id = f"{doc_id}_chunk_{i}"
            self.embeddings[chunk_id] = {
                'embedding': embedding,
                'metadata': {**metadata, 'chunk_index': i, 'total_chunks': len(chunks)}
            }
        self.save_embeddings()

    def regenerate_embeddings(self, documents_dir: str):
        print("Regenerating embeddings...")
        new_embeddings = {}
        file_queue = deque()

        for root, _, files in os.walk(documents_dir):
            for file in files:
                file_path = os.path.join(root, file)
                is_valid = True
                for x in [".git", "/data/", "/tests/", "/benchmark/"]:
                    if x in file_path:
                        is_valid = False
                        break
                for x in BANNED_EXTENSIONS:
                    if file_path.endswith(x):
                        is_valid = False
                if not is_valid:
                    continue
                assert(".ipynb" not in file_path)
                file_queue.append(file_path)

        for file_path in tqdm(file_queue, desc="Processing files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                doc_id = file_path
                chunks, encoded_chunks = self.chunk_text(content)
                for i, chunk in enumerate(chunks):
                    # print(f"{file_path}#{i}={len(encoded_chunks[i])}")
                    embedding = self.embed_text(chunk)
                    chunk_id = f"{doc_id}_chunk_{i}"
                    new_embeddings[chunk_id] = {
                        'embedding': embedding,
                        'metadata': {
                            'file_path': file_path,
                            'content': chunk,
                            'chunk_index': i,
                            'total_chunks': len(chunks)
                        }
                    }

            except Exception as err:
                logging.exception(err)
                print(f"{file_path}")

        self.embeddings = new_embeddings
        self.save_embeddings()
        print(f"Embeddings regenerated and saved to {self.embedding_file}")
