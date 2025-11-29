from langchain_nomic import NomicEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import settings
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langsmith import traceable

@traceable(run_type="embedding", name="Get Embeddings")
def get_embeddings():
    """Get Google embeddings instance."""
    return GoogleGenerativeAIEmbeddings(
        google_api_key=settings.GOOGLE_API_KEY,
        model=settings.EMBEDDING_MODEL
    )

@traceable(run_type="retriever", name="Get Vector Store")
def get_vector_store(namespace: str = None):
    """Get Pinecone vector store instance."""
    return PineconeVectorStore(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=get_embeddings(),
        pinecone_api_key=settings.PINECONE_API_KEY,
        namespace=namespace
    )

def model():
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_CHAT_MODEL,
        temperature=settings.GROQ_TEMPERATURE,
        # Limit max tokens to prevent context overflow
        max_tokens=2048,
        # Reduce timeout for faster failure detection
        timeout=60
    )