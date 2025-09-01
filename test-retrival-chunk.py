import yaml
from langchain_chroma import Chroma
from models import Models

def load_prompts(prompt_file="prompt.yaml"):
    """Load prompts from YAML file"""
    try:
        with open(prompt_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: {prompt_file} not found. Please create the prompt configuration file.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None

def format_context_with_metadata(docs):
    """Format documents with rich metadata for LLM understanding"""
    formatted_context = []
    
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        header1 = metadata.get('Header 1', '')
        header2 = metadata.get('Header 2', '')  
        header3 = metadata.get('Header 3', '')
        header_path = metadata.get('header_path', 'Root')
        source_file = metadata.get('source_file', 'Unknown')
        file_path = metadata.get('file_path', 'Unknown')
        section_hierarchy = metadata.get('section_hierarchy', '{}')
        
        doc_context = f"""
===== DOCUMENT {i} =====
FILE: {source_file}
PATH: {file_path}
SECTION_PATH: {header_path}
HEADERS: H1: "{header1}" | H2: "{header2}" | H3: "{header3}"
HIERARCHY: {section_hierarchy}

CONTENT:
{doc.page_content}
===== END DOCUMENT {i} =====
"""
        formatted_context.append(doc_context)
    
    return "\n".join(formatted_context)

def main():
    print("PathRAG - Debug Mode (Show Retrieved Chunks Only)")
    print("-" * 60)
    
    # Load prompts (not actually used here but kept for compatibility)
    prompts_config = load_prompts("prompt.yaml")
    
    # Initialize the models (only embeddings needed)
    models = Models()
    embeddings = models.embeddings_ollama
    
    # Initialize the vector store
    vector_store = Chroma(
        collection_name="markdown_documents",
        embedding_function=embeddings,
        persist_directory="./db/chroma_langchain_db"
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    print("Available commands:")
    print("  • Type your question to see retrieved chunks")
    print("  • 'quit' to exit")
    print("-" * 60)
    
    while True:
        query = input("\nUser: ").strip()
        
        if query.lower() in ['q', 'quit', 'exit']:
            print("Goodbye!")
            break
            
        if not query:
            continue
            
        try:
            print("Retrieving relevant chunks...")
            docs = retriever.get_relevant_documents(query)
            
            if docs:
                print("\nRetrieved Chunks (with Metadata):")
                print("="*50)
                print(format_context_with_metadata(docs))
            else:
                print("No documents found for this query.")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
