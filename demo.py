import os
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder, OllamaTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever, InMemoryBM25Retriever
from haystack.components.joiners import DocumentJoiner

# ==========================================
# 1. Initialize Document Store
# We need it to support BOTH Embeddings (Cosine) and Keywords
# ==========================================
document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

# ==========================================
# 2. Data Preparation
# ==========================================
raw_docs = [
    Document(content="My name is Alex, and I am an AI Engineer looking for opportunities in Germany."),
    Document(content="I specialize in Python, Haystack 2.x, and building RAG pipelines."),
    Document(content="I have strong knowledge of GDPR data privacy and local model deployment."),
    Document(content="I am currently based in Berlin and available to start immediately."),
    # Add a tricky case: Specific keyword that Embeddings might miss, but BM25 loves
    Document(content="My GitHub handle is AlexCoder99 and my reference ID is REF-2024-DE.") 
]

print("1. Indexing Documents...")

# Create Embeddings for the docs
doc_embedder = OllamaDocumentEmbedder(model="nomic-embed-text", url="http://localhost:11434")
docs_with_embeddings = doc_embedder.run(raw_docs)["documents"]

# Write to store (BM25 automatically builds its index here too)
document_store.write_documents(docs_with_embeddings)
print("   Done! Documents indexed.")

# ==========================================
# 3. Initialize Components for Hybrid Search
# ==========================================

# A. Components for Vector Search (Semantic)
text_embedder = OllamaTextEmbedder(model="nomic-embed-text", url="http://localhost:11434")
embedding_retriever = InMemoryEmbeddingRetriever(document_store=document_store)

# B. Components for Keyword Search (Exact Match)
bm25_retriever = InMemoryBM25Retriever(document_store=document_store)

# C. The Joiner (The Glue)
# This merges results from both retrievers.
# If both find the same document, it boosts its score.
document_joiner = DocumentJoiner(join_mode="concatenate", top_k=5)

# D. Prompt & LLM
template = """
Answer the question based ONLY on the following [Context].
[Context]:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ question }}
Answer:
"""
prompt_builder = PromptBuilder(template=template)
generator = OllamaGenerator(model="phi3", url="http://localhost:11434")

# ==========================================
# 4. Build the Hybrid Pipeline (Y-Shape)
# ==========================================
pipeline = Pipeline()
pipeline.add_component("text_embedder", text_embedder)
pipeline.add_component("embedding_retriever", embedding_retriever)
pipeline.add_component("bm25_retriever", bm25_retriever)
pipeline.add_component("joiner", document_joiner)
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.add_component("llm", generator)

# Connect the Logic
# Branch 1: Vector Search
pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
pipeline.connect("embedding_retriever", "joiner")

# Branch 2: Keyword Search (BM25)
pipeline.connect("bm25_retriever", "joiner")

# Merge & Generate
pipeline.connect("joiner", "prompt_builder.documents")
pipeline.connect("prompt_builder", "llm")

# ==========================================
# 5. Run it!
# ==========================================
# Try a question that requires BOTH:
# "skills" (Semantic) + "REF-2024-DE" (Exact Keyword)
question = "What is the candidate's reference ID and what does he specialize in?"
print(f"\n2. Processing Hybrid Question: '{question}'...")

result = pipeline.run({
    "text_embedder": {"text": question},       # Input for Vector Search
    "bm25_retriever": {"query": question},     # Input for Keyword Search
    "prompt_builder": {"question": question}   # Input for LLM
})

print("\n=== AI Response ===")
print(result['llm']['replies'][0])