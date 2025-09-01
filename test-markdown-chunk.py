import os
import re
from dotenv import load_dotenv
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import TextLoader
from uuid import uuid4
from datetime import datetime

load_dotenv()

def chunk_markdown_file(file_path):
    """
    Chunk a markdown file by sections (headers) and add metadata to each chunk
    """
    # Check if file exists and is markdown
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    
    if not file_path.lower().endswith(('.md', '.markdown')):
        print(f"File is not a markdown file: {file_path}")
        return []
    
    print(f"Starting to chunk markdown file: {file_path}")
    
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
                    any(line.strip().startswith(f'{i}.') for i in range(1, 100))])
        
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
            'section_hierarchy': header_levels,
            
            # Content structure analysis
            'content_types': content_types,
            'has_code': code_blocks > 0,
            'has_links': links > 0,
            'has_images': images > 0,
            'has_tables': tables > 0,
            'has_lists': lists > 0,
            
            # Element counts for PathRAG reasoning
            'code_blocks_count': code_blocks // 2,  # Each block has opening and closing
            'inline_code_count': inline_code // 2,
            'links_count': links,
            'images_count': images,
            'table_rows_estimate': tables,
            'list_items_count': lists,
            
            # PathRAG navigation context
            'is_first_chunk': i == 0,
            'is_last_chunk': i == len(md_header_splits) - 1,
            'prev_chunk_id': chunks[i-1].metadata.get('chunk_id') if i > 0 else None,
            'next_chunk_id': None,  # Will be updated after all chunks are processed
            
            # Content density and complexity
            'content_density': round(len(content.split()) / max(len(lines), 1), 2),
            'complexity_score': min(max_header_level * 0.3 + len(content_types) * 0.2 + 
                                   (code_blocks * 0.5) + (tables * 0.3), 10.0),
            
            # Search and retrieval hints
            'keywords_estimate': len(set(word.lower().strip('.,!?;:"()[]{}') 
                                       for word in content.split() 
                                       if len(word) > 3 and word.isalpha())),
            'potential_entities': [word for word in content.split() 
                                 if word[0].isupper() and len(word) > 2 and word.isalpha()][:10],
            
            # Processing metadata
            'processing_timestamp': str(os.path.getmtime(file_path)),
            'file_size_bytes': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
        })
        
        chunks.append(chunk)
    
    # Update next_chunk_id for PathRAG navigation
    for i in range(len(chunks)):
        if i < len(chunks) - 1:
            chunks[i].metadata['next_chunk_id'] = chunks[i + 1].metadata['chunk_id']
    
    return chunks

def save_chunks_to_txt(chunks, output_file="chunked_output.txt"):
    """
    Save the chunks to a text file for inspection
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== MARKDOWN CHUNKING RESULTS ===\n\n")
        f.write(f"Total chunks: {len(chunks)}\n")
        f.write("="*50 + "\n\n")
        
        for i, chunk in enumerate(chunks):
            f.write(f"CHUNK {i+1}\n")
            f.write("-" * 20 + "\n")
            f.write("METADATA:\n")
            for key, value in chunk.metadata.items():
                f.write(f"  {key}: {value}\n")
            f.write("\nCONTENT:\n")
            f.write(chunk.page_content)
            f.write("\n" + "="*50 + "\n\n")
    
    print(f"Chunks saved to: {output_file}")

def process_single_markdown(file_path, output_file=None):
    """
    Process a single markdown file and output chunks to text file
    """
    # Generate chunks
    chunks = chunk_markdown_file(file_path)
    
    if not chunks:
        print("No chunks generated.")
        return
    
    print(f"Generated {len(chunks)} chunks from the markdown file.")
    
    # Print summary to console
    print("\n=== CHUNK SUMMARY ===")
    for i, chunk in enumerate(chunks):
        headers = []
        for key, value in chunk.metadata.items():
            if key.startswith('Header'):
                headers.append(f"{key}: {value}")
        
        header_info = " | ".join(headers) if headers else "No headers"
        print(f"Chunk {i+1}: {len(chunk.page_content)} chars, {len(chunk.page_content.split())} words - {header_info}")
    
    # Save to file
    if output_file is None:
        output_file = f"chunks_{os.path.splitext(os.path.basename(file_path))[0]}.txt"
    
    save_chunks_to_txt(chunks, output_file)
    return chunks

# Example usage
if __name__ == "__main__":
    # Specify your markdown file path
    markdown_file_path = "./data/app_store/_functional.md"  # Change this to your markdown file path
    output_txt_file = "markdown.txt"   # Output file name
    
    # Process the markdown file
    chunks = process_single_markdown(markdown_file_path, output_txt_file)