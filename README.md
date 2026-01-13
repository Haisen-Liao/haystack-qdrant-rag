# 🤖 Local RAG Research Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Haystack](https://img.shields.io/badge/Haystack-2.x-orange)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-green)

## 📖 Overview

**Local RAG Research Assistant** is a full-stack, privacy-first Retrieval-Augmented Generation (RAG) application designed to help researchers and engineers interact with complex PDF documents (e.g., technical manuals, academic papers).

Unlike cloud-based solutions, this project runs **100% locally** using Ollama and Dockerized Qdrant. It ensures **GDPR compliance** and data sovereignty, making it suitable for handling sensitive or proprietary documents without sending data to external APIs.

## ✨ Key Features

* **🔒 Privacy First:** No data leaves the local machine. All inference happens via local LLMs (Phi-3 / Llama-3).
* **🧠 RAG Architecture:** Implements a state-of-the-art ETL pipeline (Extract, Transform, Load) with Semantic Search.
* **⚡ High Performance:** Uses **Qdrant** (Production-grade Vector Database) for millisecond-level retrieval.
* **🛠️ Modular Design:** Built with **Haystack 2.x**, allowing easy swapping of components (Embedders, Generators).
* **🖥️ Interactive UI:** User-friendly web interface built with **Streamlit** for file uploads and chat.

## 🏗️ Architecture

The system consists of two main pipelines managed by the Haystack orchestration framework:

```mermaid
graph TD
    subgraph "Indexing Pipeline (ETL)"
        A[PDF File] -->|PyPDFToDocument| B(Text Extraction)
        B -->|Cleaner| C(Preprocessing)
        C -->|Splitter| D(Chunking)
        D -->|Ollama Embedder| E(Vector Embedding)
        E -->|Writer| F[(Qdrant Vector DB)]
    end

    subgraph "RAG Pipeline (Inference)"
        G[User Question] -->|Ollama Embedder| H(Query Embedding)
        H -->|Retriever| F
        F -->|Top-k Documents| I(Prompt Builder)
        I -->|Context + Question| J[Local LLM (Phi-3)]
        J -->|Answer| K[Streamlit UI]
    end