import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
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

def create_prompt_template(prompts_config, prompt_type="pathrag_prompt"):
    """Create ChatPromptTemplate from YAML config"""
    if not prompts_config or prompt_type not in prompts_config:
        print(f"Error: {prompt_type} not found in prompt configuration")
        return None
    
    config = prompts_config[prompt_type]
    system_message = config["system_message"]
    human_template = config["human_template"]
    
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_template)
    ])

def format_context_with_metadata(docs):
    """Format documents with rich metadata for LLM understanding"""
    formatted_context = []
    
    for i, doc in enumerate(docs, 1):
        # Extract key metadata fields
        metadata = doc.metadata
        
        # Get header hierarchy
        header1 = metadata.get('Header 1', '')
        header2 = metadata.get('Header 2', '')  
        header3 = metadata.get('Header 3', '')
        header_path = metadata.get('header_path', 'Root')
        source_file = metadata.get('source_file', 'Unknown')
        file_path = metadata.get('file_path', 'Unknown')
        section_hierarchy = metadata.get('section_hierarchy', '{}')
        
        # Format the document with metadata
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
    print("PathRAG - Intelligent Document Retrieval System")
    print("Enhanced with YAML-based prompt configuration")
    print("-" * 60)
    
    # Load prompts from YAML
    prompts_config = load_prompts("prompt.yaml")
    if not prompts_config:
        return
    
    # Initialize the models
    models = Models()
    embeddings = models.embeddings_ollama
    llm = models.model_ollama
    
    # Initialize the vector store
    vector_store = Chroma(
        collection_name="markdown_documents",
        embedding_function=embeddings,
        persist_directory="./db/chroma_langchain_db"
    )
    
    # Create prompt template from YAML
    prompt = create_prompt_template(prompts_config, "pathrag_prompt")
    if not prompt:
        return
    
    # Create retrieval chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    print("Available commands:")
    print("  • Type your question for document search")
    print("  • 'switch api' - Switch to API-focused prompts")
    print("  • 'switch arch' - Switch to architecture-focused prompts")
    print("  • 'switch default' - Switch back to default prompts")
    print("  • 'quit' to exit")
    print("-" * 60)
    
    current_prompt_type = "pathrag_prompt"
    
    while True:
        query = input(f"\nUser ({current_prompt_type}): ").strip()
        
        if query.lower() in ['q', 'quit', 'exit']:
            print("Goodbye!")
            break
            
        # Handle prompt switching commands
        if query.lower() == 'switch api':
            prompt = create_prompt_template(prompts_config, "specialized_prompts")
            if prompt and "api_documentation" in prompts_config["specialized_prompts"]:
                api_config = prompts_config["specialized_prompts"]["api_documentation"]
                prompt = ChatPromptTemplate.from_messages([
                    ("system", api_config["system_message"]),
                    ("human", prompts_config["pathrag_prompt"]["human_template"])
                ])
                combine_docs_chain = create_stuff_documents_chain(llm, prompt)
                retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
                current_prompt_type = "api_focused"
                print("Switched to API documentation mode")
            continue
            
        elif query.lower() == 'switch arch':
            if "specialized_prompts" in prompts_config and "architecture_overview" in prompts_config["specialized_prompts"]:
                arch_config = prompts_config["specialized_prompts"]["architecture_overview"]
                prompt = ChatPromptTemplate.from_messages([
                    ("system", arch_config["system_message"]),
                    ("human", prompts_config["pathrag_prompt"]["human_template"])
                ])
                combine_docs_chain = create_stuff_documents_chain(llm, prompt)
                retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
                current_prompt_type = "architecture_focused"
                print("Switched to architecture documentation mode")
            continue
            
        elif query.lower() == 'switch default':
            prompt = create_prompt_template(prompts_config, "pathrag_prompt")
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
            current_prompt_type = "pathrag_prompt"
            print("Switched back to default mode")
            continue
            
        if not query:
            continue
            
        try:
            print("Searching...")
            result = retrieval_chain.invoke({"input": query})
            
            # Show sources first with metadata
            if result.get('context'):
                print(f"\nSources:")
                print("="*50)
                for i, doc in enumerate(result['context'], 1):
                    file_path = doc.metadata.get('source_file', 'Unknown')
                    header_path = doc.metadata.get('header_path', 'Root')
                    
                    print(f" {i}. {file_path} - {header_path}")
            
            # Use the regular result but show formatted metadata in sources
            answer = result['answer']
            
            # Clean the answer - remove thinking tags
            if '<think>' in answer and '</think>' in answer:
                answer = answer.split('</think>')[1].strip()
            elif '<think>' in answer:
                answer = answer.split('<think>')[0].strip()
            
            # Show the clean answer
            print("\nAnswer:")
            print("="*50)
            print(answer)
            
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()