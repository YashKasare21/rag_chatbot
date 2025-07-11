import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from utils import load_pdfs

# Define the path to the PDF directory and FAISS index
PDF_DIR = "data/pdfs"
FAISS_INDEX_PATH = "vectorstore/faiss_index"

def ingest_documents():
    """Ingests PDF documents, chunks them, creates embeddings, and stores them in FAISS."""
    print("Loading documents...")
    documents = load_pdfs(PDF_DIR)

    if not documents:
        print(f"No PDF documents found in {PDF_DIR}. Please add PDFs to this directory.")
        return

    print(f"Loaded {len(documents)} documents. Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    print("Creating embeddings... This may take a while.")
    # Using a local sentence-transformer model for embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Creating FAISS index...")
    # Create FAISS index from chunks and embeddings
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")

if __name__ == "__main__":
    ingest_documents()
