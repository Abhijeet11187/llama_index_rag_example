# 🦙⚡ RAG with LlamaIndex — Query Any PDF with AI

> **Ask questions. Get answers. Powered by Retrieval-Augmented Generation.**
> Built on LlamaIndex + OpenAI

---

## 📌 What is this?

This project demonstrates a complete **Retrieval-Augmented Generation (RAG)** pipeline using [LlamaIndex](https://www.llamaindex.ai/) and OpenAI's GPT models. Feed it any PDF (here, the landmark *"Attention Is All You Need"* paper), ask it anything, and get accurate, context-aware answers — without fine-tuning a single model.

RAG bridges the gap between static LLM knowledge and your private/specific documents by dynamically retrieving relevant content before generating a response.

---

## 🗺️ Pipeline Overview

```
PDF Document
     │
     ▼
[1] Data Ingestion          ← SimpleDirectoryReader loads raw text
     │
     ▼
[2] Embedding               ← OpenAI text-embedding-3-small converts text → vectors
     │
     ▼
[3] Vector Indexing         ← VectorStoreIndex stores & indexes embeddings
     │
     ▼
[4] Retrieval               ← Semantic search fetches relevant chunks for your query
     │
     ▼
[5] Response Generation     ← GPT-4o-mini synthesizes a final grounded answer
```

---

## 🚀 Features

- **Zero-boilerplate RAG** — LlamaIndex handles chunking, embedding, and indexing automatically
- **PDF ingestion** via `SimpleDirectoryReader` — drop in any PDF, no preprocessing needed
- **Semantic search** using OpenAI's `text-embedding-3-small` for fast, accurate retrieval
- **LLM-powered responses** with GPT-4o-mini for cost-efficient, high-quality answers
- **Node inspection** — examine retrieved chunks, metadata, and scores at every step
- **Plug-and-play query engine** — one-liner querying after setup

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| RAG Framework | `llama-index==0.14.0` |
| Embedding Model | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Vector Store | `VectorStoreIndex` (in-memory) |
| Document Loader | `SimpleDirectoryReader` |
| Response Synthesizer | `get_response_synthesizer` |
| Runtime | Python / Google Colab |

---

## 📂 Project Structure

```
rag_with_llamaIndex.ipynb   # Main notebook — full RAG pipeline
attention_all_you_need.pdf  # Sample input document (the Transformer paper)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-username/rag-with-llamaindex.git
cd rag-with-llamaindex
```

### 2. Install dependencies
```bash
pip install llama-index==0.14.0
pip install llama-index-readers-file
pip install llama-index-readers-web
pip install unstructured
```

### 3. Set your OpenAI API key
```python
from getpass import getpass
OPENAI_API_KEY = getpass("Enter your OpenAI API key: ")
```

### 4. Run the notebook
Open `rag_with_llamaIndex.ipynb` in Jupyter or Google Colab and run all cells top to bottom.

---

## 💡 Usage

### Full pipeline in ~10 lines

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Load document
document = SimpleDirectoryReader(input_files=['attention_all_you_need.pdf']).load_data()

# Set up models
llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

# Index and query
index = VectorStoreIndex.from_documents(document, embed_model=embed_model)
query_engine = index.as_query_engine(llm=llm)

print(query_engine.query("What is self-attention?").response)
```

### Sample queries used in the notebook

```python
query_engine.query("What is self attention in the attention is all you need paper")
query_engine.query("How are Q, K, V calculated in the attention is all you need paper")
query_engine.query("What activation function is used in the attention is all you need paper")
```

---

## 🔍 Deep Dive: Pipeline Stages

### Stage 1 — Data Ingestion
`SimpleDirectoryReader` loads your PDF and returns a list of `Document` objects. Each object contains the raw text and metadata (filename, page number, etc.).

```python
document = SimpleDirectoryReader(input_files=['your_file.pdf']).load_data()
document[0].text      # raw text
document[0].metadata  # page info, source, etc.
```

### Stage 2 — Embedding
Text chunks are converted into high-dimensional vectors using OpenAI's embedding model. These vectors capture semantic meaning, enabling similarity-based search.

```python
embedding_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
```

### Stage 3 — Vector Indexing
LlamaIndex stores all embeddings in a `VectorStoreIndex`. For small PDFs, LlamaIndex handles chunking automatically — no manual splitting needed.

```python
index = VectorStoreIndex.from_documents(document, embed_model=embedding_model)
```

### Stage 4 — Retrieval
The retriever performs semantic search, returning the most relevant text nodes for any query.

```python
retriever = index.as_retriever()
nodes = retriever.retrieve("What is multi-head attention?")
nodes[0].text      # retrieved chunk
nodes[0].metadata  # source metadata
```

### Stage 5 — Response Generation
The response synthesizer passes retrieved context + your query to the LLM, which generates a grounded, accurate answer.

```python
from llama_index.core import get_response_synthesizer
response_synthesizer = get_response_synthesizer(llm=llm)
query_engine = index.as_query_engine(llm=llm, response_synthesizer=response_synthesizer)
response = query_engine.query("Explain the transformer architecture")
print(response.response)
```

---

## 📄 Sample Document

The notebook uses the **"Attention Is All You Need"** paper (Vaswani et al., 2017) — the foundational paper that introduced the Transformer architecture. You can swap it for any PDF of your choice.

---

## 🔄 LangChain vs LlamaIndex

| Feature | LangChain | LlamaIndex |
|---|---|---|
| Vector Store | ChromaDB | VectorStoreIndex |
| Focus | General LLM orchestration | Document-centric RAG |
| Auto chunking | Manual setup | Built-in for common formats |
| Best for | Agents, chains, tools | Document Q&A, RAG pipelines |

---

## 🧠 Key Concepts

- **RAG (Retrieval-Augmented Generation)** — Combines retrieval of relevant documents with LLM generation to produce factually grounded responses
- **Embeddings** — Dense vector representations of text that capture semantic meaning
- **Vector Index** — A data structure enabling fast similarity search over embeddings
- **Node** — LlamaIndex's unit of retrieved content, containing text + metadata + score
- **Query Engine** — LlamaIndex's end-to-end interface: retrieves context and generates a response in one call

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a PR for:
- Support for additional document formats (DOCX, web pages, etc.)
- Persistent vector storage (ChromaDB, Pinecone, Weaviate)
- Evaluation metrics for retrieval quality
- UI/chat interface wrapper

---



## 📄 License

This project is for educational purposes. Feel free to use and adapt it for your own learning.
