import streamlit as st
from rag_chain import RAGPipeline
import os

# Initialize the RAG pipeline
@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline()

rag_pipeline = get_rag_pipeline()

st.title("RAG Chatbot for Internal Documents")

# User input
query = st.text_input("Ask a question about your documents:")

if query:
    with st.spinner("Searching for answers..."):
        answer, sources = rag_pipeline.run(query)
        st.write("**Answer:**", answer)

        if sources:
            st.subheader("Source Documents:")
            for source in sources:
                st.write(f"- {os.path.basename(source.metadata['source'])} (Page {source.metadata['page']})")
                with st.expander("View Snippet"):
                    st.write(source.page_content)
        else:
            st.info("No specific source documents found for this query.")
