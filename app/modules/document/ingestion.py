from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.core.agent import get_vector_store
from pinecone import Pinecone

def process_document(file_path: str, user_id: str):
    print(f"process_document: Starting with user_id={user_id}, file_path={file_path}")
    
    # 1. Load the document
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    print(f"process_document: Loaded {len(documents)} documents")

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = text_splitter.split_documents(documents)
    print(f"process_document: Split into {len(docs)} chunks")

    # Add user_id to metadata for each document chunk
    for doc in docs:
        doc.metadata['user_id'] = user_id

    # 3. Clear existing vectors and create new vector store
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    existing_indexes = pc.list_indexes()
    index_exists = any(idx.name == settings.PINECONE_INDEX_NAME for idx in existing_indexes.indexes or [])
    if not index_exists:
        available = [idx.name for idx in existing_indexes.indexes or []]
        raise ValueError(f"Pinecone index '{settings.PINECONE_INDEX_NAME}' does not exist. Available indexes: {available}. Please create the index in your Pinecone dashboard first.")
    
    index = pc.Index(settings.PINECONE_INDEX_NAME)
    print(f"process_document: Clearing namespace={user_id}")
    try:
        index.delete(delete_all=True, namespace=user_id)
    except Exception as e:
        if '404' in str(e) or 'Not Found' in str(e) or 'Namespace not found' in str(e):
            pass  # Namespace is empty, proceed
        else:
            raise

    # 4. Store the new vectors in Pinecone
    print(f"process_document: Storing {len(docs)} docs in namespace={user_id}")
    vector_store = get_vector_store(namespace=user_id)
    vector_store.add_documents(docs)
    print(f"process_document: Successfully stored documents in namespace={user_id}")

def clear_user_vectors(user_id: str):
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    existing_indexes = pc.list_indexes()
    index_exists = any(idx.name == settings.PINECONE_INDEX_NAME for idx in existing_indexes.indexes or [])
    if not index_exists:
        return  # Index does not exist, nothing to clear

    index = pc.Index(settings.PINECONE_INDEX_NAME)
    try:
        index.delete(delete_all=True, namespace=user_id)
    except Exception as e:
        if '404' in str(e) or 'Not Found' in str(e) or 'Namespace not found' in str(e):
            pass  # Namespace is empty, nothing to clear
        else:
            raise



    