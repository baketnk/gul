import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import time

from transformers import EarlyStoppingCallback
from transformers import AutoTokenizer, AutoModel,AutoModelForMaskedLM,  AdamW, get_linear_schedule_with_warmup, DataCollatorForLanguageModeling
import json
import os
from typing import List, Dict
import re
from tqdm import tqdm

from .embed_base import BaseEmbeddingModel

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def calculate_mrr(similarities):
    sorted_indices = np.argsort(similarities)[::-1]
    for rank, index in enumerate(sorted_indices, 1):
        if index == 0:  # Assuming the first document is always relevant
            return 1 / rank
    return 0


class JinaEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_path: str = './models/jina-embeddings-v2-base-code'):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.mlm = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)
        # Load the model directly using AutoModel
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model.to(self.device)


    def fine_tune(self, documents: List[Dict], epochs: int = 10, batch_size: int = 8, log_every: int = 10,
              learning_rate: float = 5e-6, warmup_steps: int = 0, early_stopping_patience: int = 3,
              max_length: int = 512, accumulation_steps: int = 4):
        dataset = DocumentChunkDataset(documents, self, max_length=max_length)
        
        # Split into train and validation sets
        train_size = int(0.9 * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.mlm_collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.mlm_collate_fn)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        # Early stopping
        early_stopping = EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        best_val_loss = float('inf')
        patience = early_stopping.early_stopping_patience
        no_improve_epochs = 0

        self.model.train()
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_loss = 0
            self.model.train()
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accumulation_steps

                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * accumulation_steps

                if (batch_idx + 1) % log_every == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_dataloader)}, Loss: {loss.item() * accumulation_steps:.4f}")

            avg_train_loss = epoch_loss / len(train_dataloader)
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
        
            print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_duration:.2f} seconds. Average Training Loss: {avg_train_loss:.4f}")


            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Validation Loss: {avg_val_loss:.4f}")

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve_epochs = 0
                print("Saving best model...")
                self.model.save_pretrained('./models/fine_tuned_jina_embeddings_ft')
                self.tokenizer.save_pretrained('./models/fine_tuned_jina_embeddings_ft')
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                print(f"Early stopping triggered. No improvement for {patience} epochs.")
                break

        print("Fine-tuning completed.")


    def mlm_collate_fn(self, batch):
        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)
        return collator(batch)

    def embed_text(self, text: str) -> list[float]:
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to(self.device)
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # use hidden_states instead of last_hidden_state
            token_embeddings = outputs.hidden_states[-1]
            
            # mean pooling
            attention_mask = inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            
            # normalize
            embeddings = F.normalize(mean_pooled, p=2, dim=1)
            
        return embeddings.squeeze().cpu().tolist()


def load_documents(directory: str) -> list[dict]:
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
                except exception as e:
                    print(f"error reading file {file_path}: {e}")
    
    return documents


class DocumentChunkDataset(Dataset):
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
