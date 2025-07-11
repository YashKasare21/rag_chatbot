import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

def load_pdfs(directory: str) -> list[Document]:
    """Loads all PDF documents from a specified directory."""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            print(f"Loading {filepath}")
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            for doc in docs:
                # Add source and page to metadata
                doc.metadata["source"] = filepath
                doc.metadata["page"] = doc.metadata.get("page", "N/A") # Ensure page is present
            documents.extend(docs)
    return documents
