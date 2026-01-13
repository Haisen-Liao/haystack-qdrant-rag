from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

# 1. Connect to the Existing Document Store
# Note: 'recreate_index' is NOT used here because we want to read existing data.
document_store = QdrantDocumentStore(
    url="http://localhost:6333",
    index="my_paper_db", 
    embedding_dim=768
)

# 2. Initialize Components
# The embedder is used here to convert the USER QUERY into a vector.
text_embedder = OllamaTextEmbedder(model="nomic-embed-text", url="http://localhost:11434")
retriever = QdrantEmbeddingRetriever(document_store=document_store, top_k=3)
generator = OllamaGenerator(model="phi3", url="http://localhost:11434", timeout=300)

# 3. Define Prompt Template
# This instructs the LLM on how to behave.
template = """
You are an academic research assistant. Please answer the user's question based ONLY on the following [Context].
Be professional and accurate. If possible, cite specific details from the context.

[Context]:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

User Question: {{ question }}
Answer:
"""
prompt_builder = PromptBuilder(template=template)

# 4. Build the Query Pipeline
pipeline = Pipeline()
pipeline.add_component("text_embedder", text_embedder)
pipeline.add_component("retriever", retriever)
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.add_component("llm", generator)

# Connect the flow: Query -> Vector -> Retriever -> Prompt -> LLM
pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
pipeline.connect("retriever", "prompt_builder.documents")
pipeline.connect("prompt_builder", "llm")

# 5. Interactive Loop
# Allows the user to keep asking questions without restarting the script.
while True:
    user_input = input("\nEnter your question about the paper (or type 'q' to quit): ")
    if user_input.lower() == 'q':
        break
    
    print("AI is reading and thinking...")
    result = pipeline.run({
        "text_embedder": {"text": user_input},
        "prompt_builder": {"question": user_input}
    })
    
    print("--------------------------------------------------")
    print(result['llm']['replies'][0])
    print("--------------------------------------------------")