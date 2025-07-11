import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Define paths
FAISS_INDEX_PATH = "vectorstore/faiss_index"
LLM_MODEL_PATH = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

class RAGPipeline:
    def __init__(self):
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = self._load_vectorstore()
        self.llm = self._load_llm()
        self.qa_chain = self._setup_qa_chain()

    def _load_vectorstore(self):
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. "
                "Please run `python ingest.py` first to create the vector store."
            )
        print(f"Loading FAISS index from {FAISS_INDEX_PATH}")
        return FAISS.load_local(FAISS_INDEX_PATH, self.embeddings, allow_dangerous_deserialization=True)

    def _load_llm(self):
        if not os.path.exists(LLM_MODEL_PATH):
            raise FileNotFoundError(
                f"LLM model not found at {LLM_MODEL_PATH}. "
                "Please download 'mistral-7b-instruct-v0.1.Q4_K_M.gguf' and place it in the 'models' directory."
            )
        print(f"Loading LLM from {LLM_MODEL_PATH}")
        # Adjust model_type and config based on your GGUF model
        return CTransformers(
            model=LLM_MODEL_PATH,
            model_type="mistral", # or "llama", "gpt2", etc. depending on your model
            config={'max_new_tokens': 512, 'temperature': 0.1, 'gpu_layers': 0}
        )

    def _setup_qa_chain(self):
        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else.
        Helpful answer:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Create a RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff", # "stuff" combines all documents into a single prompt
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}), # Retrieve top 5 relevant documents
            return_source_documents=True, # Return the source documents used for the answer
            chain_type_kwargs={"prompt": PROMPT}
        )
        return qa_chain

    def run(self, query: str):
        print(f"Running query: {query}")
        result = self.qa_chain.invoke({"query": query})
        answer = result["result"]
        source_documents = result.get("source_documents", [])
        return answer, source_documents

if __name__ == "__main__":
    # Example usage:
    try:
        rag_pipeline = RAGPipeline()
        # Test query
        test_query = "What is the policy on vacation days?"
        answer, sources = rag_pipeline.run(test_query)
        print("\nAnswer:", answer)
        print("\nSources:")
        for source in sources:
            print(f"- {os.path.basename(source.metadata['source'])} (Page {source.metadata['page']})")
            print(f"  Snippet: {source.page_content[:200]}...")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have run `python ingest.py` and downloaded the LLM model.")
