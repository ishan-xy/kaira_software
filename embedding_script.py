import os
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

# Load BGE small embedding model
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

def read_markdown_files(folder_path: str) -> List[str]:
    """Read all markdown files and return list of contents"""
    all_content = []
    for file in Path(folder_path).glob('*.md'):
        with open(file, 'r', encoding='utf-8') as f:
            all_content.append(f.read())
    return all_content

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

def process_folder(folder_path: str, chunk_size: int = 512, overlap: int = 50):
    """Process all markdown files and create embeddings"""
    # Read all files
    print(f"Reading markdown files from: {folder_path}")
    files_content = read_markdown_files(folder_path)
    print(f"Found {len(files_content)} markdown files")
    
    # Chunk all content
    all_chunks = []
    for content in files_content:
        chunks = chunk_text(content, chunk_size, overlap)
        all_chunks.extend(chunks)
    
    print(f"Created {len(all_chunks)} chunks")
    
    # Generate embeddings
    embeddings = model.encode(
        all_chunks,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    return embeddings, all_chunks

def main():
    # Default to the `documents` folder next to this script
    default_docs = Path(__file__).parent / "documents"
    folder_path = str(default_docs)

    if not default_docs.exists():
        print(f"Error: documents folder not found at {default_docs}")
        return

    embeddings, chunks = process_folder(folder_path)

    # Save for later use (saved to current working directory)
    np.save('embeddings.npy', embeddings)
    np.save('chunks.npy', chunks)

    print("Done!")


if __name__ == '__main__':
    main()
