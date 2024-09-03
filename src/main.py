import argparse
import sys
from typing import List
import os
import yaml
from .importers import import_url
from .vectorsearch import VectorSearch 
from pathlib import Path
from .config import config

def search(query: str) -> List[dict]:
    vector_search = VectorSearch()
    return vector_search.search(query)


def finetune(args):
   raise NotImplementedError()

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
            vector_search = VectorSearch()
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
            vector_search = VectorSearch()
            documents_dir = config.get('document_store').get('local_directory')
            vector_search.regenerate_embeddings(documents_dir)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        else:
            parser.print_help()
            sys.exit(1)

if __name__ == "__main__":
    main()
