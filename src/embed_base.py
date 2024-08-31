

import re
from typing import List, Dict

class BaseEmbeddingModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.setup_model()

    def setup_model(self):
        raise NotImplementedError("Subclasses must implement this method")

    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError("Subclasses must implement this method")

    def chunk_text(self, text: str, max_length: int = 512) -> List[str]:
        chunks = []
        current_chunk = ""
        current_chunk_tokens = 0

        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            sentence_tokens = self.tokenizer.tokenize(sentence)
            sentence_token_count = len(sentence_tokens)

            if current_chunk_tokens + sentence_token_count <= max_length:
                current_chunk += " " + sentence
                current_chunk_tokens += sentence_token_count
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_chunk_tokens = sentence_token_count

            # If the current sentence is longer than max_length, split it
            while current_chunk_tokens > max_length:
                token_limit = max_length - 1  # Leave room for [SEP] token
                truncated_chunk = self.tokenizer.decode(self.tokenizer.encode(current_chunk)[:token_limit])
                chunks.append(truncated_chunk)
                current_chunk = current_chunk[len(truncated_chunk):]
                current_chunk_tokens = len(self.tokenizer.tokenize(current_chunk))

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks


    def fine_tune(self, documents: List[Dict], **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
