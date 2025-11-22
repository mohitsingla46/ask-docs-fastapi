from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nomic import NomicEmbeddings
from app.core.config import settings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

def process_document(file_path: str, user_id: str):
    # 1. Load the document
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = text_splitter.split_documents(documents)

    # Add user_id to metadata for each document chunk
    for doc in docs:
        doc.metadata['user_id'] = user_id

    # 3. Generate embeddings for each chunk
    embeddings = NomicEmbeddings(
        nomic_api_key=settings.NOMIC_API_KEY,
        model=settings.NOMIC_EMBEDDING_MODEL
    )

    # 4. Clear existing vectors and create new vector store
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    existing_indexes = pc.list_indexes()
    index_exists = any(idx.name == settings.PINECONE_INDEX_NAME for idx in existing_indexes.indexes or [])
    if not index_exists:
        available = [idx.name for idx in existing_indexes.indexes or []]
        raise ValueError(f"Pinecone index '{settings.PINECONE_INDEX_NAME}' does not exist. Available indexes: {available}. Please create the index in your Pinecone dashboard first.")
    
    index = pc.Index(settings.PINECONE_INDEX_NAME)
    try:
        index.delete(delete_all=True, namespace=user_id)
    except Exception as e:
        if '404' in str(e) or 'Not Found' in str(e) or 'Namespace not found' in str(e):
            pass  # Namespace is empty, proceed
        else:
            raise

    PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=settings.PINECONE_INDEX_NAME,
        pinecone_api_key=settings.PINECONE_API_KEY,
        namespace=user_id
    )




    