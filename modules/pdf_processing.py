# Cell code 1 - Pdf Files Folder Processing Pipeline

import os
import json
import PyPDF2
from tqdm import tqdm  # For progress tracking
from typing import List, Dict, Any

# ============= CONFIGURATION SETTINGS =============
# Set your folder paths here
PDF_FOLDER = "Files/Pdf_Files"  # Directory containing your PDF files
OUTPUT_FILE = "data/processed/all_pdf_chunks.json"  # Name of the output JSON file
CHUNK_SIZE = 1000  # Target size of each chunk in characters
CHUNK_OVERLAP = 200  # Overlap between chunks in characters
# ================================================

def ensure_directory_exists(file_path):
    """Create directory if it doesn't exist."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

# Define the chunking function
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into overlapping chunks of approximately chunk_size characters."""
    # If text is shorter than chunk_size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Find the end of the current chunk
        end = start + chunk_size
        
        # If we're at the end of the text, add the final chunk and break
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to find a good breaking point (e.g., end of sentence or paragraph)
        breakpoint = text.rfind('.', start + chunk_size // 2, end)
        if breakpoint == -1:
            breakpoint = text.rfind('?', start + chunk_size // 2, end)
        if breakpoint == -1:
            breakpoint = text.rfind('\n', start + chunk_size // 2, end)
        if breakpoint == -1:
            # If no good breakpoint, just use the maximum length
            breakpoint = end
        else:
            # Include the breaking character
            breakpoint += 1
        
        # Add the chunk
        chunks.append(text[start:breakpoint])
        
        # Move to next chunk with overlap
        start = breakpoint - chunk_overlap
    
    return chunks

def process_pdf(file_path, chunk_size=1000, chunk_overlap=200):
    """Process a PDF file with improved error handling and metadata extraction."""
    all_chunks = []
    
    try:
        # Open the PDF file
        with open(file_path, 'rb') as file:
            try:
                # Create a PDF reader object
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Get document info
                info = pdf_reader.metadata
                title = info.title if info and info.title else os.path.basename(file_path)
                author = info.author if info and info.author else "Unknown"
                
                # Get total number of pages
                num_pages = len(pdf_reader.pages)
                
                # Process each page
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    
                    # Try to extract text
                    try:
                        text = page.extract_text()
                    except Exception as e:
                        print(f"Warning: Could not extract text from page {page_num+1} in {file_path}: {e}")
                        continue
                    
                    # Skip empty pages
                    if not text.strip():
                        print(f"Warning: Page {page_num+1} in {file_path} appears to be empty or contains non-extractable content")
                        continue
                    
                    # Create text chunks using the global chunk_text function
                    text_chunks = chunk_text(text, chunk_size, chunk_overlap)
                    
                    # Create chunks with metadata
                    for i, chunk_content in enumerate(text_chunks):
                        chunk = {
                            "text": chunk_content,
                            "metadata": {
                                "source": file_path,
                                "title": title,
                                "author": author,
                                "page_number": page_num + 1,
                                "total_pages": num_pages,
                                "chunk_number": i + 1,
                                "total_chunks_in_page": len(text_chunks),
                                "document_type": "pdf"
                            }
                        }
                        
                        all_chunks.append(chunk)
                
                print(f"Successfully processed PDF: {file_path}")
                print(f"Extracted {len(all_chunks)} chunks from {num_pages} pages")
                
            except Exception as e:
                print(f"Error processing PDF {file_path}: {e}")
                
    except Exception as e:
        print(f"Error opening file {file_path}: {e}")
    
    return all_chunks

def process_pdf_folder(folder_path, output_file="pdf_chunks.json", chunk_size=1000, chunk_overlap=200):
    """Process all PDF files in a folder and save the chunks to a JSON file."""
    all_chunks = []
    pdf_files_count = 0
    total_chunks_count = 0
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return []
    
    # Get all PDF files in the folder
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in folder '{folder_path}'.")
        return []
    
    print(f"Found {len(pdf_files)} PDF files in '{folder_path}'.")
    
    # Process each PDF file
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(folder_path, pdf_file)
        chunks = process_pdf(pdf_path, chunk_size, chunk_overlap)
        
        if chunks:  # If chunks were successfully extracted
            all_chunks.extend(chunks)
            pdf_files_count += 1
            total_chunks_count += len(chunks)
    
    # Save all chunks to a JSON file
    if all_chunks:
        try:
            # Ensure directory exists
            ensure_directory_exists(output_file)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            print(f"\nSuccessfully saved {total_chunks_count} chunks from {pdf_files_count} PDF files to '{output_file}'")
        except Exception as e:
            print(f"Error saving chunks to JSON file: {e}")
    
    return all_chunks

def process_pdfs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process PDF files from specified paths in the configuration
    
    Args:
        config (Dict): Configuration dictionary with PDF folder paths
    
    Returns:
        Dict: Processed PDF document chunks with summary statistics
    """
    # Use paths from config or default to PDF_FOLDER
    pdf_folder = config.get('pdf_paths', ['Files/Pdf_Files'])[0]
    output_file = config.get('output_file', 'data/processed/all_pdf_chunks.json')
    chunk_size = config.get('chunk_size', 1000)
    chunk_overlap = config.get('chunk_overlap', 200)
    
    # Process PDF folder
    chunks = process_pdf_folder(
        pdf_folder, 
        output_file,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Prepare summary
    summary = {
        'total_chunks': len(chunks),
        'sample_chunk': chunks[0] if chunks else None,
        'chunks': chunks
    }
    
    return summary
    
if __name__ == "__main__":
    config = {
        'pdf_paths': ['Files/Pdf_Files'],
        'output_file': 'data/processed/all_pdf_chunks.json',
        'chunk_size': 1000,
        'chunk_overlap': 200
    }
    result = process_pdfs(config)
    
    # Print summary statistics
    print(f"Total Chunks: {result['total_chunks']}")
    
    # Print sample chunk if available
    if result['sample_chunk']:
        print("\nSample Chunk:")
        print(f"Source: {result['sample_chunk']['metadata']['source']}")
        print(f"Page: {result['sample_chunk']['metadata']['page_number']}")
        print(f"Text preview: {result['sample_chunk']['text'][:150]}...")
        print("\nMetadata:")
        for key, value in result['sample_chunk']['metadata'].items():
            print(f"  {key}: {value}")