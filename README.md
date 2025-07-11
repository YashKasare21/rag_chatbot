# RAG Chatbot for Internal Documents

This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions based on internal company documents (PDFs). It leverages open-source models and frameworks to provide a locally deployable solution with a Streamlit-based user interface.

## Features

- **PDF Document Ingestion**: Processes PDF documents, extracts text, and chunks it for efficient retrieval.
- **Local Embeddings**: Uses `SentenceTransformer` (`all-MiniLM-L6-v2`) for generating document embeddings locally.
- **FAISS Vector Store**: Stores document embeddings and metadata in a FAISS index for fast similarity search.
- **Local Large Language Model (LLM)**: Integrates a local GGUF-formatted LLM (Mistral-7B-Instruct-v0.1) for generating answers.
- **Streamlit UI**: Provides an intuitive web interface for interacting with the chatbot.
- **Source Attribution**: Displays the source PDF and page number for retrieved information.
- **CPU-Optimized**: Configured to run the LLM on the CPU, suitable for systems without powerful GPUs.

## Project Structure

```
rag_chatbot/
├── app.py                  # Streamlit web application for the chatbot UI
├── ingest.py               # Script for ingesting PDF documents and creating the FAISS index
├── rag_chain.py            # Defines the RAG pipeline, including LLM and retrieval logic
├── utils.py                # Helper functions for PDF loading and text processing
├── requirements.txt        # Python dependencies
├── data/
│   └── pdfs/               # Directory to place your PDF documents
│       └── your_document.pdf
├── models/
│   └── mistral-7b-instruct-v0.1.Q4_K_M.gguf # Downloaded LLM model
└── vectorstore/
    └── faiss_index/        # Directory where the FAISS index will be stored
        ├── index.faiss
        └── index.pkl
```

## Setup and Installation

Follow these steps to set up and run the RAG chatbot on your local machine.

### 1. Clone the Repository (if applicable)

If you received this project as a repository, clone it:

```bash
git clone <repository_url>
cd rag_chatbot
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv env
# On Windows:
.\env\Scripts\activate
# On macOS/Linux:
source env/bin/activate
```

### 3. Install Dependencies

Install all required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Download the LLM Model

Download the `mistral-7b-instruct-v0.1.Q4_K_M.gguf` model. You can typically find this model on Hugging Face (e.g., from TheBloke's GGUF models). Place the downloaded `.gguf` file into the `models/` directory. Create the `models/` directory if it doesn't exist.

Example download (replace with actual download method if different):

```bash
# You might use a tool like `huggingface-cli download` or download manually
# Example (conceptual, actual command may vary):
# huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.1-GGUF mistral-7b-instruct-v0.1.Q4_K_M.gguf --local-dir models
```

### 5. Place Your PDF Documents

Create a `data/pdfs/` directory inside your project root if it doesn't exist. Place all the PDF documents you want the chatbot to learn from into this directory.

Example:

```
rag_chatbot/
└── data/
    └── pdfs/
        ├── USA_Employee_Handbook-Freely_Available.pdf
        └── another_document.pdf
```

### 6. Ingest Documents and Create Vector Store

Run the ingestion script to process your PDFs, chunk their content, generate embeddings, and create the FAISS vector store. This step can take some time depending on the number and size of your documents.

```bash
python ingest.py
```

Upon successful completion, a `vectorstore/faiss_index/` directory will be created, containing `index.faiss` and `index.pkl`.

## Usage

Once the setup is complete and documents are ingested, you can run the Streamlit application.

```bash
streamlit run app.py
```

This command will open the RAG Chatbot interface in your default web browser (usually `http://localhost:8501`).

Type your questions into the input box and press Enter. The chatbot will retrieve relevant information from your documents and provide an answer, along with the source documents and snippets.

### Example Questions

- "What is the policy on vacation days?"
- "How many sick leaves are allowed?"
- "What are the company's values?"
- "Can you tell me about the employee benefits?"

## Troubleshooting

- **`ModuleNotFoundError`**: If you encounter a `ModuleNotFoundError` for `langchain`, `langchain-community`, `pypdf`, or any other package, ensure all dependencies are installed:
  ```bash
  pip install -r requirements.txt
  ```
  If a specific module like `pypdf` is still missing, try installing it directly:
  ```bash
  pip install pypdf
  ```

- **`FileNotFoundError: FAISS index not found`**: This means the vector store has not been created. Run the ingestion script:
  ```bash
  python ingest.py
  ```

- **`FileNotFoundError: LLM model not found`**: Ensure the `mistral-7b-instruct-v0.1.Q4_K_M.gguf` file is correctly placed in the `models/` directory.

- **Slow Responses / High GPU Usage**: The LLM is configured to run on the CPU (`gpu_layers: 0`). If you still experience slow responses, consider:
    - Using a smaller LLM model.
    - Reducing the `k` value in the retriever (fewer documents to process).
    - Optimizing your system resources.

- **No Specific Source Documents Found**: Ensure your PDFs contain the information relevant to your queries. Verify that the document loading, chunking, and embedding processes are working correctly by checking the console output of `ingest.py`.

## Performance Evaluation (Manual)

After running the chatbot, you can manually evaluate its performance based on the following criteria:

- **Relevance**: Are the answers accurate and directly derived from the content of your PDFs?
- **Source Attribution**: Does the chatbot correctly identify and display the source PDF filename and page number for its answers?
- **Response Time**: How quickly does the chatbot generate responses? This can be a subjective measure but helps in understanding the user experience.
- **Handling Out-of-Context Questions**: How does the chatbot respond to questions that are not covered by your documents? Ideally, it should state that it doesn't know the answer rather than generating incorrect or fabricated information (hallucinating).

## Further Enhancements (Future Work)

- **Advanced Chunking Strategies**: Experiment with different chunking methods (e.g., semantic chunking) to improve retrieval accuracy.
- **Hybrid Search**: Combine keyword search with vector similarity search for more robust retrieval.
- **Evaluation Metrics**: Implement automated evaluation metrics (e.g., RAGAS) to quantitatively assess the chatbot's performance.
- **User Feedback**: Add a mechanism for users to provide feedback on answer quality.
- **Scalability**: For larger document sets, consider more robust vector databases (e.g., ChromaDB, Pinecone) and distributed processing.
- **Fine-tuning LLM**: Fine-tune a smaller LLM on your specific document domain for potentially better performance.