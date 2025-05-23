from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os

from dotenv import load_dotenv
os.environ.setdefault("USER_AGENT", "MyAppBot/1.0")
load_dotenv()


class RAGManager:
    def __init__(self, vector_store_path: str = "vectorstore"):
        """Initialize the RAG manager.

        Args:
            base_url: The base URL to scrape data from
            vector_store_path: Path where the vector store will be saved
        """
        self.vector_store_path = vector_store_path
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0
        )
        self.vectorstore = None

    def load_or_create_vectorstore(self, force_reload: bool = False) -> None:
        """Load the existing vector store or create a new one if it doesn't exist."""
        vs_file = f"{self.vector_store_path}/faiss_store.faiss"
        pkl_file = f"{self.vector_store_path}/faiss_store.pkl"
        print("Checking for vector store files...")
        print(f"vs_file path: {vs_file} - Exists: {os.path.exists(vs_file)}")
        print(f"pkl_file path: {pkl_file} - Exists: {os.path.exists(pkl_file)}")
        print(f"force_reload: {force_reload}")

        # Check if vector store exists and we're not forcing a reload
        if os.path.exists(vs_file) and os.path.exists(pkl_file) and not force_reload:
            print("Loading existing vector store...")
            self.vectorstore = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                "faiss_store",
                allow_dangerous_deserialization=True
            )
        else:
            print("Creating new vector store...")
            self._create_vectorstore()

    def _create_vectorstore(self) -> None:
        """Create a new vector store from web content."""
        # Load and process website content
        loader = PyPDFLoader("pdf/boleto_merged.pdf")
        documents = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)

        # Create a vector store
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)

        # Create a directory if it doesn't exist
        os.makedirs(self.vector_store_path, exist_ok=True)

        # Save the vector store
        self.vectorstore.save_local(self.vector_store_path, "faiss_store")
        print(f"Vector store saved to {self.vector_store_path}")

    def query(self, query: str, k: int = 4) -> str:
        """Query the RAG system.

        Args:
            query: The query string
            k: Number of relevant documents to retrieve

        Returns:
            Generated response based on retrieved documents
        """
        if not self.vectorstore:
            self.load_or_create_vectorstore()

        # Search for relevant documents
        docs = self.vectorstore.similarity_search(query, k=k)

        # Prepare context from documents
        context = "\n\n".join([doc.page_content for doc in docs])

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that provides accurate information based on the given context.
            Use the context to answer the question. If the context doesn't contain enough information,
            say so, but try to provide relevant information from what is available."""),
            ("human", "Context:\n{context}\n\nQuestion: {query}")
        ])

        # Generate response
        chain = prompt | self.llm
        response = chain.invoke({
            "context": context,
            "query": query
        })

        return response.content

def main():
    # Example usage
    rag = RAGManager()

    # Force reload if needed
    rag.load_or_create_vectorstore()

    # Example queries
    queries = [
        "What is your name?",
    ]

    for query in queries:
        print(f"\nQuestion: {query}")
        response = rag.query(query)
        print(f"Answer: {response}")

if __name__ == "__main__":
    main()
 