import pytest
from unittest.mock import patch, MagicMock


from langchain_community.vectorstores import FAISS

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag import RAGManager



# ---------- Test Initialization ----------
def test_ragmanager_initialization():
    rag = RAGManager(vector_store_path="test_vectorstore")
    assert rag.vector_store_path == "test_vectorstore"
    assert rag.vectorstore is None


# ---------- Test Vectorstore Load/Create ----------

def test_load_existing_vectorstore(monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda path: True)

    def fake_load_local(path, embeddings, index_name, allow_dangerous_deserialization=False):
        class DummyStore:
            def similarity_search(self, query, k): return []
        return DummyStore()

    monkeypatch.setattr(FAISS, "load_local", fake_load_local)

    rag = RAGManager()
    rag.load_or_create_vectorstore()
    assert rag.vectorstore is not None


def test_create_vectorstore(monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda path: False)

    dummy_docs = [{"page_content": "Test page content"}]

    class DummyLoader:
        def load(self): return dummy_docs

    class DummySplitter:
        def split_documents(self, docs): return docs

    class DummyStore:
        @staticmethod
        def from_documents(docs, embeddings): return DummyStore()
        def save_local(self, path, name): pass
        def similarity_search(self, query, k): return []

    monkeypatch.setattr("your_module_file.PyPDFLoader", lambda path: DummyLoader())  # Replace with real filename
    monkeypatch.setattr("your_module_file.RecursiveCharacterTextSplitter", lambda *args, **kwargs: DummySplitter())
    monkeypatch.setattr("your_module_file.FAISS", MagicMock(from_documents=DummyStore.from_documents))

    rag = RAGManager()
    rag._create_vectorstore()
    assert rag.vectorstore is not None


# ---------- Test Query Logic ----------

def test_query_triggers_vectorstore_creation(monkeypatch):
    called = {"load_or_create": False}

    rag = RAGManager()

    def mock_load_or_create_vectorstore():
        called["load_or_create"] = True
        class DummyStore:
            def similarity_search(self, query, k): return [{"page_content": "Test content"}]
        rag.vectorstore = DummyStore()

    rag.load_or_create_vectorstore = mock_load_or_create_vectorstore

    class DummyLLM:
        def invoke(self, inputs): return type("Obj", (object,), {"content": "Dummy answer"})

    monkeypatch.setattr("your_module_file.ChatPromptTemplate.from_messages", lambda msgs: lambda inputs: DummyLLM())

    result = rag.query("What is your name?")
    assert result == "Dummy answer"
    assert called["load_or_create"]


def test_query_with_no_relevant_docs(monkeypatch):
    rag = RAGManager()
    rag.vectorstore = type("Dummy", (), {"similarity_search": lambda self, query, k: []})()

    class DummyLLM:
        def invoke(self, inputs): return type("Obj", (object,), {"content": "Not enough information."})

    monkeypatch.setattr("your_module_file.ChatPromptTemplate.from_messages", lambda msgs: lambda inputs: DummyLLM())

    result = rag.query("Unknown question?")
    assert result == "Not enough information."


# ---------- Error Handling ----------

def test_pdf_load_failure(monkeypatch):
    class FailingLoader:
        def load(self): raise FileNotFoundError("PDF not found")

    monkeypatch.setattr("your_module_file.PyPDFLoader", lambda path: FailingLoader())

    rag = RAGManager()
    with pytest.raises(FileNotFoundError):
        rag._create_vectorstore()


# ---------- Integration ----------

@pytest.mark.skip(reason="Requires actual PDF and Groq/HuggingFace integration")
def test_end_to_end_query():
    rag = RAGManager()
    rag.load_or_create_vectorstore(force_reload=True)
    response = rag.query("What is boleto?")
    assert isinstance(response, str)
    assert len(response) > 0
