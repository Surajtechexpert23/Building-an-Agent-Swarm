# 🔁 Agent Swarm Implementation

A sophisticated multi-agent swarm system using **LangChain** and **LangGraph** to orchestrate cooperative AI agents for customer support and knowledge management. Designed specifically for **InfinitePay's** financial services.

---

## 🚀 Project Overview

This project implements a network of specialized agents collaborating intelligently to manage support queries, retrieve information, and maintain tone consistency across conversations.

---

## 🧠 Core Agents

### 1. 🧭 Router Agent (`agents/router.py`)

* Central dispatcher and coordinator  
* Routes queries to specialized agents  
* Maintains state and workflow history

### 2. 📚 Knowledge Agent (`agents/knowledge.py`)

* Product and service information via RAG  
* Uses FAISS vector store  
* Handles tech docs and company content

### 3. 💬 Support Agent (`agents/support.py`)

* Support tickets, payments, refunds  
* Call scheduling and issue tracking  
* FAQ handling

### 4. 🎭 Personality Agent (`agents/personality.py`)

* Adds consistent tone, style, and formatting  
* Maintains contextual voice  
* Transforms plain responses

---

## 🏗️ Architecture

### 🔧 Tech Stack

| Component         | Tool/Library          |
| ----------------- | --------------------- |
| API Server        | FastAPI               |
| LLM Orchestration | LangChain + LangGraph |
| Vector Store      | FAISS                 |
| PDF Parsing       | PyPDF                 |
| Embeddings        | Sentence Transformers |
| Inference         | Groq                  |
| Frontend          | Streamlit Dashboard   |

---

## 📁 Directory Structure

```bash
.
├── agents/                # Agent implementations
│   ├── router.py
│   ├── knowledge.py
│   ├── support.py
│   └── personality.py
├── pdf/                   # PDF docs (company materials)
├── vectorstore/           # FAISS vector indices
├── tools/                 # Utility modules (RAG, support tools)
├── api.py                 # FastAPI server
├── graph.py               # LangGraph workflow
├── state.py               # State & types
├── node.py                # Agent base node
└── requirements.txt
````

---

## 🔑 API Key Setup

### 🌐 Groq API

* Go to: [Groq Console](https://console.groq.com/)
* Create account & generate API key

### 🔍 Tavily API

* Go to: [Tavily AI](https://app.tavily.com/)
* Sign up and visit the **API** section
* Generate your API Key
---

## 🛠️ Installation & Setup

### 🔗 Prerequisites

* Python 3.10+
* Git
* pip
* 8GB+ RAM
* SSD recommended for FAISS

### 💻 Install Steps

```bash
# Clone repo
git clone https://github.com/yourusername/agent_swarm_implementation.git
cd agent_swarm_implementation

# Create virtual environment
python -m venv venv
source venv/bin/activate    # On Linux/Mac
# OR
venv\Scripts\activate       # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 📄 Environment Setup

Create a `.env` file:

```env

GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key

# Paths
VECTORSTORE_PATH=./vectorstore
PDF_PATH=./pdf

# Server
HOST=0.0.0.0
PORT=8000
```


## 🐳 Docker Setup

To containerize and run the app using Docker:

### 🛠 Build Docker Image

```bash
docker build -t agent-swarm-app .
```

### 🚀 Run Container

```bash
docker run -p 8000:8000 agent-swarm-app
```

---

## 🧪 Running the System

### Start API Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Optional: Launch Streamlit Dashboard

```bash
streamlit run dashboard.py
```

### Access Interfaces

* API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
* Streamlit UI: [http://localhost:8501](http://localhost:8501)

---

## 🧩 Usage Examples

### Basic Query

```python
from graph import invoke_graph

response = invoke_graph("What services does InfinitePay offer?")
print(response["response"])
```

### Support Request

```python
response = invoke_graph("I'm having issues with my payment terminal")
print(response["response"])
```

---

## 📡 API Endpoints

### `POST /query`

```json
{
  "message": "How do I integrate your API?"
}
```

**Response**

```json
{
  "messages": [
    {
      "content": "Here's how to integrate our API...",
      "type": "AIMessage"
    }
  ],
  "error": null
}
```

### `GET /health`

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

## ⚙️ Vector Store Management

### Build Index

```bash
python scripts/build_vectorstore.py
```

### Update Documents

```bash
python scripts/update_docs.py
```

---

## 🧪 Troubleshooting

| Issue              | Fix                                          |
| ------------------ | -------------------------------------------- |
| Vector store error | Check if FAISS is built and PDFs are present |
| API down           | Verify `.env` keys and ports                 |
| Memory crash       | Lower batch size, upgrade RAM                |

### Logs

* `logs/api.log`
* `logs/agents.log`
* `logs/error.log`

---

## 📈 Roadmap

* [ ] 🌐 Multi-language support
* [ ] 📊 Advanced analytics dashboard
* [ ] 🧱 Custom agent development kit
* [ ] 🔗 Integrate additional LLM providers

