import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict
import os
from tqdm import tqdm

from .embed_base import BaseEmbeddingModel

class MxbaiEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_path: str = './models/mxbai-embed-large-v1'):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)

    def transform_query(self, query: str) -> str:
        return f'Represent this sentence for searching relevant passages: {query}'

    def pooling(self, outputs: torch.Tensor, inputs: Dict, strategy: str = 'cls') -> np.ndarray:
        if strategy == 'cls':
            outputs = outputs[:, 0]
        elif strategy == 'mean':
            outputs = torch.sum(
                outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"], dim=1, keepdim=True)
        else:
            raise NotImplementedError
        return outputs.detach().cpu().numpy()

    def embed_text(self, text: str) -> List[float]:
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to(self.device)
            outputs = self.model(**inputs).last_hidden_state
            embeddings = self.pooling(outputs, inputs, 'cls')
        return embeddings.squeeze().tolist()

    def search(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
        query_embedding = self.embed_text(self.transform_query(query))
        doc_embeddings = [self.embed_text(doc) for doc in documents]

        similarities = cos_sim([query_embedding], doc_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'document': documents[idx],
                'score': float(similarities[idx])
            })

        return results

    def fine_tune(self, documents: List[Dict], epochs: int = 10, batch_size: int = 8, log_every: int = 10,
                  learning_rate: float = 5e-6, warmup_steps: int = 0, early_stopping_patience: int = 3,
                  max_length: int = 512, accumulation_steps: int = 4):
        # Fine-tuning is not implemented for this model
        print("Fine-tuning is not implemented for the MXBAI model.")

def load_documents(directory: str) -> List[Dict]:
    documents = []
    text_extensions = ('.txt', '.md', '.py', '.js', '.java', '.cpp', '.html', '.css', '.lua')
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(text_extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    documents.append({
                        'text': content,
                        'metadata': {
                            'file_path': file_path,
                            'file_type': os.path.splitext(file)[1][1:]
                        }
                    })
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    
    return documents

def calculate_mrr(similarities):
    sorted_indices = np.argsort(similarities)[::-1]
    for rank, index in enumerate(sorted_indices, 1):
        if index == 0:  # Assuming the first document is always relevant
            return 1 / rank
    return 0

class DocumentChunkDataset:
    def __init__(self, documents: List[Dict], embedding_model: BaseEmbeddingModel, max_length: int = 512):
        self.chunks = []
        self.embedding_model = embedding_model
        self.max_length = max_length

        for doc in tqdm(documents):
            text = doc['text']
            chunks = self.embedding_model.chunk_text(text, max_length)
            for chunk in chunks:
                self.chunks.append({
                    'text': chunk,
                    'metadata': doc.get('metadata', {})
                })

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        encoded = self.embedding_model.tokenizer.encode_plus(
            chunk['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze()
        }
