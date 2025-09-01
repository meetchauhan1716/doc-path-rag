import os
import re
import time
from dotenv import load_dotenv
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from uuid import uuid4
from datetime import datetime
from models import Models

load_dotenv()

# Initialize the models
models = Models()
embeddings = models.embeddings_ollama
llm = models.model_ollama

# Define constants
data_folder = "./data"
check_interval = 10  # seconds

# Chroma vector store setup
vector_store = Chroma(
    collection_name="markdown_documents",
    embedding_function=embeddings,
    persist_directory="./db/chroma_langchain_db"  # where to save the data locally
)

def find_all_markdown_files(root_folder):
    """
    Recursively find all markdown files in nested folders
    """
    markdown_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.md', '.markdown')) and not file.startswith('_'):
                file_path = os.path.join(root, file)
                markdown_files.append(file_path)
    return markdown_files

def save_file_list(markdown_files, output_file="file_list.txt"):
    """
    Save the list of markdown files to a text file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== MARKDOWN FILES FOUND ===\n\n")
        f.write(f"Total files: {len(markdown_files)}\n")
        f.write("="*50 + "\n\n")
        
        for i, file_path in enumerate(markdown_files, 1):
            relative_path = os.path.relpath(file_path, data_folder)
            file_size = os.path.getsize(file_path)
            f.write(f"{i:3d}. {relative_path}\n")
            f.write(f"     Full path: {file_path}\n")
            f.write(f"     Size: {file_size:,} bytes\n\n")
    
    print(f"File list saved to: {output_file}")

def chunk_markdown_file(file_path):
    """
    Chunk a markdown file by sections (headers) and add metadata to each chunk
    """
    # Check if file exists and is markdown
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    
    print(f"Starting to chunk markdown file: {file_path}")
    
    try:
        # Load the markdown file
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        
        # Define headers to split on
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"), 
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]
        
        # Create markdown header splitter
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False  # Keep headers in the content
        )
        
        # Split the document
        md_header_splits = markdown_splitter.split_text(documents[0].page_content)
        
        # Add additional metadata to each chunk
        chunks = []
        for i, chunk in enumerate(md_header_splits):
            # Generate unique ID for each chunk
            chunk_id = str(uuid4())
            
            # Extract content analysis
            content = chunk.page_content
            lines = content.split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            
            # Count different markdown elements
            code_blocks = content.count('```')
            inline_code = content.count('`') - code_blocks * 2
            links = content.count('[') and content.count('](')
            images = content.count('![')
            tables = content.count('|')
            lists = len([line for line in lines if line.strip().startswith(('- ', '* ', '+ ')) or 
                        any(line.strip().startswith(f'{j}.') for j in range(1, 100))])
            
            # Determine content type
            content_types = []
            if code_blocks > 0:
                content_types.append('code')
            if links > 0:
                content_types.append('links')
            if images > 0:
                content_types.append('images')
            if tables > 0:
                content_types.append('tables')
            if lists > 0:
                content_types.append('lists')
            if not content_types:
                content_types.append('text')
            
            # Extract header hierarchy path
            header_path = []
            header_levels = {}
            for key, value in chunk.metadata.items():
                if key.startswith('Header'):
                    level = int(key.split(' ')[1])
                    header_levels[level] = value
            
            # Build hierarchical path
            if header_levels:
                sorted_levels = sorted(header_levels.keys())
                header_path = [header_levels[level] for level in sorted_levels]
            
            # Determine section depth and context
            max_header_level = max(header_levels.keys()) if header_levels else 0
            min_header_level = min(header_levels.keys()) if header_levels else 0
            
            # Calculate readability metrics
            sentences = len([s for s in content.split('.') if s.strip()])
            avg_words_per_sentence = len(content.split()) / max(sentences, 1)
            
            # Add comprehensive metadata
            chunk.metadata.update({
                # Basic identifiers
                'chunk_id': chunk_id,
                'chunk_index': i,
                'source_file': os.path.basename(file_path),
                'file_path': file_path,
                'relative_path': os.path.relpath(file_path, data_folder),
                'folder_path': os.path.dirname(os.path.relpath(file_path, data_folder)),
                
                # Content metrics
                'chunk_length': len(content),
                'word_count': len(content.split()),
                'line_count': len(lines),
                'non_empty_lines': len(non_empty_lines),
                'sentence_count': sentences,
                'avg_words_per_sentence': round(avg_words_per_sentence, 2),
                
                # PathRAG specific - Hierarchical context
                'header_path': ' > '.join(header_path),
                'header_depth': len(header_path),
                'max_header_level': max_header_level,
                'min_header_level': min_header_level,
                'section_hierarchy': str(header_levels),  # Convert to string for Chroma
                
                # Content structure analysis
                'content_types': ','.join(content_types),  # Convert to string for Chroma
                'has_code': code_blocks > 0,
                'has_links': links > 0,
                'has_images': images > 0,
                'has_tables': tables > 0,
                'has_lists': lists > 0,
                
                # Element counts for PathRAG reasoning
                'code_blocks_count': code_blocks // 2,
                'inline_code_count': inline_code // 2,
                'links_count': links,
                'images_count': images,
                'table_rows_estimate': tables,
                'list_items_count': lists,
                
                # PathRAG navigation context
                'is_first_chunk': i == 0,
                'is_last_chunk': i == len(md_header_splits) - 1,
                'prev_chunk_id': chunks[i-1].metadata.get('chunk_id') if i > 0 else '',
                'next_chunk_id': '',  # Will be updated after all chunks are processed
                
                # Content density and complexity
                'content_density': round(len(content.split()) / max(len(lines), 1), 2),
                'complexity_score': round(min(max_header_level * 0.3 + len(content_types) * 0.2 + 
                                           (code_blocks * 0.5) + (tables * 0.3), 10.0), 2),
                
                # Search and retrieval hints
                'keywords_estimate': len(set(word.lower().strip('.,!?;:"()[]{}') 
                                           for word in content.split() 
                                           if len(word) > 3 and word.isalpha())),
                'potential_entities': ','.join([word for word in content.split() 
                                             if word[0].isupper() and len(word) > 2 and word.isalpha()][:10]),
                
                # Processing metadata
                'processing_timestamp': datetime.now().isoformat(),
                'file_size_bytes': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            })
            
            chunks.append(chunk)
        
        # Update next_chunk_id for PathRAG navigation
        for i in range(len(chunks)):
            if i < len(chunks) - 1:
                chunks[i].metadata['next_chunk_id'] = chunks[i + 1].metadata['chunk_id']
        
        return chunks
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return []

def ingest_markdown_file(file_path):
    """
    Ingest a single markdown file into ChromaDB
    """
    print(f"Starting to ingest markdown file: {file_path}")
    
    # Generate chunks using our enhanced strategy
    chunks = chunk_markdown_file(file_path)
    
    if not chunks:
        print(f"No chunks generated for file: {file_path}")
        return
    
    # Prepare documents and IDs for ChromaDB
    documents = chunks
    uuids = [chunk.metadata['chunk_id'] for chunk in chunks]
    
    print(f"Adding {len(documents)} chunks to the vector store.")
    try:
        vector_store.add_documents(documents=documents, ids=uuids)
        print(f"File {file_path} ingested successfully with {len(chunks)} chunks.")
    except Exception as e:
        print(f"Error adding documents to vector store: {str(e)}")

def ingest_all_markdown_files():
    """
    Find and ingest all markdown files from the data folder
    """
    print(f"Scanning for markdown files in: {data_folder}")
    
    # Find all markdown files
    markdown_files = find_all_markdown_files(data_folder)
    
    if not markdown_files:
        print("No markdown files found!")
        return
    
    print(f"Found {len(markdown_files)} markdown files")
    
    # Save file list
    save_file_list(markdown_files)
    
    # Process each file
    successful_ingestions = 0
    failed_ingestions = 0
    
    for i, file_path in enumerate(markdown_files, 1):
        print(f"\n--- Processing file {i}/{len(markdown_files)} ---")
        try:
            ingest_markdown_file(file_path)
            successful_ingestions += 1
            
            # Rename file to mark as processed (add _ prefix)
            directory = os.path.dirname(file_path)
            filename = os.path.basename(file_path)
            new_filename = "_" + filename
            new_file_path = os.path.join(directory, new_filename)
            os.rename(file_path, new_file_path)
            print(f"Marked file as processed: {new_filename}")
            
        except Exception as e:
            print(f"Failed to process file {file_path}: {str(e)}")
            failed_ingestions += 1
    
    print(f"\n=== INGESTION COMPLETE ===")
    print(f"Successful: {successful_ingestions}")
    print(f"Failed: {failed_ingestions}")
    print(f"Total files: {len(markdown_files)}")

def main_loop():
    """
    Main loop to continuously monitor for new markdown files
    """
    print("Starting markdown ingestion monitoring...")
    
    while True:
        try:
            # Find unprocessed markdown files (not starting with _)
            markdown_files = find_all_markdown_files(data_folder)
            
            if markdown_files:
                print(f"\nFound {len(markdown_files)} unprocessed markdown files")
                
                for file_path in markdown_files:
                    ingest_markdown_file(file_path)
                    
                    # Mark as processed
                    directory = os.path.dirname(file_path)
                    filename = os.path.basename(file_path)
                    new_filename = "_" + filename
                    new_file_path = os.path.join(directory, new_filename)
                    os.rename(file_path, new_file_path)
                    print(f"Marked file as processed: {new_filename}")
            else:
                print("No new markdown files found")
            
            print(f"Waiting {check_interval} seconds before next check...")
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\nStopping ingestion monitoring...")
            break
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            time.sleep(check_interval)

# Example usage
if __name__ == "__main__":
    print("Markdown to ChromaDB Ingestion Tool")
    print("="*40)
    
    choice = input("Choose mode:\n1. Process all files once\n2. Continuous monitoring\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        ingest_all_markdown_files()
    elif choice == "2":
        main_loop()
    else:
        print("Invalid choice. Running one-time processing...")
        ingest_all_markdown_files()