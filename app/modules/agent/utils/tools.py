from langchain.tools import tool
from app.core.agent import get_vector_store
from app.core.agent import model

@tool
def search_documents(query: str, user_id: str) -> str:
    """Search documents for relevant information based on the query."""
    try:
        print(f"search_documents: Searching with user_id={user_id}, query={query}")
        
        # Get vector store for the user
        vectorstore = get_vector_store(namespace=user_id)
        print(f"search_documents: Vector store created for namespace={user_id}")
        
        # Perform similarity search
        docs = vectorstore.similarity_search(query, k=5)
        print(f"search_documents found {len(docs)} documents for query: {query}")
        
        # Debug: Try searching without namespace to see if documents exist
        vectorstore_all = get_vector_store(namespace=None)
        all_docs = vectorstore_all.similarity_search(query, k=5)
        print(f"search_documents: Found {len(all_docs)} documents in default namespace")
        
        if not docs:
            return "No relevant documents found for the query."
        
        # Format results
        results = []
        for i, doc in enumerate(docs, 1):
            results.append(f"Result {i}:\n{doc.page_content}\n")
        
        return "\n".join(results)
    
    except Exception as e:
        print(f"search_documents error: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error searching documents: {str(e)}"
    
tools = [search_documents]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model().bind_tools(tools)