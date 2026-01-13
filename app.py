import streamlit as st
import os
import tempfile
from pathlib import Path

# Haystack Components
from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder, OllamaTextEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(page_title="My AI Research Assistant", layout="wide")
st.title("🤖 Local AI Research Assistant (Powered by Haystack & Qdrant)")

# Initialize Session State (for storing chat history)
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 2. Cache Resources (Key Optimization)
# Use @st.cache_resource to ensure model and DB connections are initialized only once
# ==========================================

@st.cache_resource
def get_document_store():
    # Connect to Qdrant running in Docker
    return QdrantDocumentStore(
        url="http://localhost:6333",
        index="my_paper_db",
        embedding_dim=768,
        # recreate_index=False means we keep existing data, don't wipe it every time
        recreate_index=False 
    )

@st.cache_resource
def get_indexing_pipeline(_document_store):
    """Create Indexing Pipeline (Process PDFs)"""
    pipeline = Pipeline()
    pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component("cleaner", DocumentCleaner(remove_empty_lines=True, remove_extra_whitespaces=True))
    pipeline.add_component("splitter", DocumentSplitter(split_by="word", split_length=250, split_overlap=50))
    pipeline.add_component("embedder", OllamaDocumentEmbedder(model="nomic-embed-text", url="http://localhost:11434"))
    pipeline.add_component("writer", DocumentWriter(document_store=_document_store))

    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")
    return pipeline

@st.cache_resource
def get_rag_pipeline(_document_store):
    """Create RAG Pipeline (Question Answering)"""
    template = """
    You are an academic research assistant. Please answer the user's question based ONLY on the following [Context].
    Be professional and accurate. If possible, cite specific details.

    [Context]:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    User Question: {{ question }}
    Answer:
    """
    
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", OllamaTextEmbedder(model="nomic-embed-text", url="http://localhost:11434"))
    # Note: Keep top_k=5 and use phi3 for speed
    pipeline.add_component("retriever", QdrantEmbeddingRetriever(document_store=_document_store, top_k=3))
    pipeline.add_component("prompt_builder", PromptBuilder(template=template))
    pipeline.add_component("llm", OllamaGenerator(model="phi3", url="http://localhost:11434", timeout=360)) 

    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")
    return pipeline

# Load resources
doc_store = get_document_store()
indexing_pipeline = get_indexing_pipeline(doc_store)
rag_pipeline = get_rag_pipeline(doc_store)

# ==========================================
# 3. Sidebar: File Upload
# ==========================================
with st.sidebar:
    st.header("📂 Upload New Paper")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Start Indexing"):
            with st.spinner("Parsing, splitting, and indexing into Vector DB..."):
                # Streamlit uploads are in-memory; we need to save to a temp file for Haystack to read
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # Run Indexing Pipeline
                indexing_pipeline.run({"converter": {"sources": [Path(tmp_file_path)]}})
                
                # Delete temporary file
                os.remove(tmp_file_path)
                
            st.success("✅ Paper successfully indexed! You can now ask questions.")

# ==========================================
# 4. Main Interface: Chat Window
# ==========================================

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("What do you want to ask about the paper?"):
    # 1. Display user question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call Haystack RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("AI is reading and thinking..."):
            result = rag_pipeline.run({
                "text_embedder": {"text": prompt},
                "prompt_builder": {"question": prompt}
            })
            response = result['llm']['replies'][0]
            st.markdown(response)
    
    # 3. Save AI response
    st.session_state.messages.append({"role": "assistant", "content": response})