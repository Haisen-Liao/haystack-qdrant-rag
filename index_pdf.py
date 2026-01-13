import os
from pathlib import Path
from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder

# ==========================================
# 1. Define File Paths
# ==========================================
# Ensure there is a file named "paper.pdf" in your current directory.
# You can use a list to process multiple files if needed.
pdf_file_path = [Path("./NIPS-2017-attention-is-all-you-need-Paper.pdf")]

# ==========================================
# 2. Initialize Qdrant Document Store
# ==========================================
document_store = QdrantDocumentStore(
    url="http://localhost:6333",
    index="my_paper_db",  # We create a specific index for research papers
    recreate_index=True,  # True = Overwrite previous data (Good for testing)
    embedding_dim=768,    # Must match the model dimension (nomic-embed-text)
    similarity="cosine"
)

# ==========================================
# 3. Initialize Components
# ==========================================

# A. Converter: specifically designed to extract text from PDF files
converter = PyPDFToDocument()

# B. Cleaner: PDF extraction often contains noise (headers, footers, page numbers)
# This component helps clean up the text before processing.
cleaner = DocumentCleaner(
    remove_empty_lines=True,
    remove_extra_whitespaces=True,
    remove_repeated_substrings=False
)

# C. Splitter: Research papers are long, so we must chunk them.
# 'respect_sentence_boundary' ensures we don't cut a sentence in half.
splitter = DocumentSplitter(
    split_by="word",
    split_length=250,    
    split_overlap=50,
    respect_sentence_boundary=True 
)

# D. Embedder: Converts text chunks into vectors
embedder = OllamaDocumentEmbedder(model="nomic-embed-text", url="http://localhost:11434")

# E. Writer: Writes the processed documents into the Qdrant database
writer = DocumentWriter(document_store=document_store)

# ==========================================
# 4. Build the Indexing Pipeline
# ==========================================
pipeline = Pipeline()
pipeline.add_component("converter", converter)
pipeline.add_component("cleaner", cleaner)
pipeline.add_component("splitter", splitter)
pipeline.add_component("embedder", embedder)
pipeline.add_component("writer", writer)

# Connect the components in logical order
pipeline.connect("converter", "cleaner")
pipeline.connect("cleaner", "splitter")
pipeline.connect("splitter", "embedder")
pipeline.connect("embedder", "writer")

# ==========================================
# 5. Run the Pipeline (Ingestion)
# ==========================================
print(f"Processing file: {pdf_file_path} ...")

# Note: The converter expects a dictionary with a 'sources' key
pipeline.run({"converter": {"sources": pdf_file_path}})

print("Success! The paper has been indexed into 'my_paper_db'.")