import argparse
import sys
from typing import List
import os
import yaml
from .importers import import_url
from .vectorsearch import VectorSearch 
from .embed_base import BaseEmbeddingModel
from .embed import JinaEmbeddingModel
from .embed_mxbai import MxbaiEmbeddingModel
from pathlib import Path
from .config import config

ORIGINAL_MODEL_PATH = './models/jina-embeddings-v2-base-code'
FINE_TUNED_MODEL_PATH = './models/fine_tuned_jina_embeddings_ft'

DEBUG_MODEL=False

def get_embedding_model(use_original=False, use_mxbai=False):
    if use_mxbai:
        return MxbaiEmbeddingModel('./models/mxbai-embed-large-v2')
    if use_original:
        return JinaEmbeddingModel(ORIGINAL_MODEL_PATH)
    if os.path.exists(FINE_TUNED_MODEL_PATH):
        return JinaiEmbeddingModel(FINE_TUNED_MODEL_PATH)
    raise FileNotFoundError("Fine-tuned model not found. Please run the 'finetune' command first.")

def search(query: str) -> List[dict]:
    model = get_embedding_model(use_original=DEBUG_MODEL)
    vector_search = VectorSearch(model)
    return vector_search.search(query)


def finetune(args):
    if not config:
        print("Error: Could not load configuration.")
        sys.exit(1)

    documents_dir = config.get('document_store').get('local_directory')
    documents = load_documents(documents_dir)

    print("Fine-tuning the model...")
    original_model = JinaEmbeddingModel(ORIGINAL_MODEL_PATH)
    fine_tuned_model.fine_tune(
        documents,
        epochs=args.epochs,
        batch_size=args.batch_size,
        log_every=args.log_every,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        early_stopping_patience=args.early_stopping_patience,
        max_length=args.max_length,
        accumulation_steps=args.accumulation_steps
    )
    print("Fine-tuning completed.")
    print(f"Fine-tuned model saved to: {FINE_TUNED_MODEL_PATH}")

def main():
    parser = argparse.ArgumentParser(description="Document processing and search tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Import URL command
    import_parser = subparsers.add_parser("import", help="Import content from a URI")
    import_parser.add_argument("uri", type=str, help="URL to import")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search the vector database")
    search_parser.add_argument("query", type=str, help="Search query")

    regenerate_parser = subparsers.add_parser("regenerate", help="Regenerate embeddings for all documents")

    # Fine-tune command
    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune the embedding model")
    finetune_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    finetune_parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    finetune_parser.add_argument("--log-every", type=int, default=10, help="Log progress every N batches")
    finetune_parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate for fine-tuning")
    finetune_parser.add_argument("--warmup-steps", type=int, default=0, help="Number of warmup steps for learning rate scheduler")
    finetune_parser.add_argument("--early-stopping-patience", type=int, default=3, help="Patience for early stopping")
    finetune_parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length for tokenization")
    finetune_parser.add_argument("--accumulation-steps", type=int, default=4, help="Number of gradient accumulation steps")

    args = parser.parse_args()

    if args.command == "import":
        try:
            model = get_embedding_model()
            vector_search = VectorSearch(model)
            import_url(args.uri, vector_search)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif args.command == "search":
        try:
            results = search(args.query)
            for result in results:
                print(result)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif args.command == "finetune":
        finetune(args)
    elif args.command == "regenerate":
        try:
            model = get_embedding_model(use_original=DEBUG_MODEL)
            vector_search = VectorSearch(model)
            documents_dir = Path(config.get('document_store').get('local_directory'))
            vector_search.regenerate_embeddings(documents_dir)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        else:
            parser.print_help()
            sys.exit(1)

if __name__ == "__main__":
    main()
