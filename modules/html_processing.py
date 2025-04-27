# modules/html_processing.py
import os
import json
import logging
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from tqdm import tqdm

# ============= CONFIGURATION SETTINGS =============
# Users can modify these settings easily
HTML_PROCESSING_CONFIG = {
    # Main processing paths
    'HTML_FOLDER': "Files/Html_Files",  # Directory containing HTML files
    'OUTPUT_FILE': "data/processed/all_html_chunks.json",  # Output JSON file path
    
    # Chunking parameters
    'CHUNK_SIZE': 1000,  # Target size of each chunk in characters
    'CHUNK_OVERLAP': 200,  # Overlap between chunks in characters
    
    # Logging configuration
    'LOG_LEVEL': logging.INFO,
    'LOG_FORMAT': '%(asctime)s - %(levelname)s: %(message)s',
    'LOG_DATE_FORMAT': '%Y-%m-%d %H:%M:%S',
    
    # File processing options
    'ALLOWED_EXTENSIONS': ['.html', '.htm'],  # Supported HTML file extensions
    'ENCODING': 'utf-8',  # Default file encoding
    'ERROR_HANDLING': 'replace',  # How to handle encoding errors
    
    # Parsing options
    'REMOVE_ELEMENTS': ['script', 'style', 'meta', 'noscript'],  # Elements to remove
    'PARSER': 'html.parser',  # BeautifulSoup parser to use
}

# Configure logging based on settings
logging.basicConfig(
    level=HTML_PROCESSING_CONFIG['LOG_LEVEL'], 
    format=HTML_PROCESSING_CONFIG['LOG_FORMAT'],
    datefmt=HTML_PROCESSING_CONFIG['LOG_DATE_FORMAT']
)

def ensure_directory_exists(file_path: str) -> None:
    """Create directory if it doesn't exist."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Simplified breakpoint finding
        breakpoint = end
        for pos in [
            text.rfind('.', start + chunk_size // 2, end),
            text.rfind('?', start + chunk_size // 2, end),
            text.rfind('\n', start + chunk_size // 2, end)
        ]:
            if pos != -1:
                breakpoint = pos + 1
                break
        
        chunks.append(text[start:breakpoint])
        start = breakpoint - chunk_overlap
    
    return chunks

def process_html(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Process HTML files with configurable settings
    
    Args:
        config (Dict, optional): Override default configuration
    
    Returns:
        Dict: Processed HTML document chunks with summary statistics
    """
    # Merge provided config with default config
    processing_config = HTML_PROCESSING_CONFIG.copy()
    if config:
        processing_config.update(config)
    
    html_folder = processing_config['HTML_FOLDER']
    output_file = processing_config['OUTPUT_FILE']
    chunk_size = processing_config['CHUNK_SIZE']
    chunk_overlap = processing_config['CHUNK_OVERLAP']
    
    # Ensure output directory exists
    ensure_directory_exists(output_file)
    
    # Check if directory exists
    if not os.path.exists(html_folder):
        logging.error(f"Directory {html_folder} does not exist")
        return {'total_chunks': 0, 'chunks': []}
    
    # Find HTML files
    html_files = [
        f for f in os.listdir(html_folder) 
        if any(f.lower().endswith(ext) for ext in processing_config['ALLOWED_EXTENSIONS'])
    ]
    
    if not html_files:
        logging.warning(f"No HTML files found in {html_folder}")
        return {'total_chunks': 0, 'chunks': []}
    
    logging.info(f"Found {len(html_files)} HTML files in '{html_folder}'")
    
    all_chunks = []
    
    # Process each HTML file
    for html_file in tqdm(html_files, desc="Processing HTML files"):
        html_path = os.path.join(html_folder, html_file)
        
        try:
            with open(html_path, 'r', encoding=processing_config['ENCODING'], errors=processing_config['ERROR_HANDLING']) as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, processing_config['PARSER'])
            
            # Remove unnecessary elements
            for element in soup(processing_config['REMOVE_ELEMENTS']):
                element.decompose()
            
            title = soup.title.string if soup.title else "Untitled HTML Document"
            text = soup.get_text(separator=' ', strip=True)
            
            # Chunk the text
            text_chunks = chunk_text(text, chunk_size, chunk_overlap)
            
            # Create chunk objects
            for i, chunk_content in enumerate(text_chunks):
                chunk = {
                    "text": chunk_content,
                    "metadata": {
                        "source": html_path,
                        "source_name": html_file,
                        "title": title,
                        "chunk_number": i + 1,
                        "total_chunks": len(text_chunks),
                        "document_type": "html"
                    }
                }
                all_chunks.append(chunk)
            
            logging.info(f"Processed {html_file}: {len(text_chunks)} chunks")
        
        except Exception as e:
            logging.error(f"Error processing {html_file}: {e}")
    
    # Save chunks to JSON
    if all_chunks:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved {len(all_chunks)} chunks to {output_file}")
        except Exception as e:
            logging.error(f"Error saving chunks: {e}")
    
    # Return summary
    return {
        'total_chunks': len(all_chunks),
        'sample_chunk': all_chunks[0] if all_chunks else None,
        'chunks': all_chunks
    }

# Optional: main block for standalone testing
if __name__ == "__main__":
    result = process_html()
    
    print(f"Total Chunks: {result['total_chunks']}")
    
    if result['sample_chunk']:
        print("\nSample Chunk:")
        print(f"Source: {result['sample_chunk']['metadata']['source']}")
        print(f"Text preview: {result['sample_chunk']['text'][:150]}...")