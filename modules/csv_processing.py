# modules/csv_processing.py
import os
import json
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm

# ============= CONFIGURATION SETTINGS =============
CSV_PROCESSING_CONFIG = {
    # Main processing paths
    'CSV_FOLDER': "Files/Csv_Files",
    'OUTPUT_FILE': "data/processed/all_csv_chunks.json",
    
    # Processing parameters
    'TEXT_COLUMNS': None,  # Auto-detect if None
    'ID_COLUMN': None,     # Use row index if None
    'CHUNK_SIZE': 5,       # Rows per chunk
    
    # Logging configuration
    'LOG_LEVEL': logging.INFO,
    'LOG_FORMAT': '%(asctime)s - %(levelname)s: %(message)s',
    'LOG_DATE_FORMAT': '%Y-%m-%d %H:%M:%S',
    
    # File processing options
    'ENCODING': 'utf-8',
    'ERROR_HANDLING': 'replace'
}

# Configure logging
logging.basicConfig(
    level=CSV_PROCESSING_CONFIG['LOG_LEVEL'], 
    format=CSV_PROCESSING_CONFIG['LOG_FORMAT'],
    datefmt=CSV_PROCESSING_CONFIG['LOG_DATE_FORMAT']
)

def ensure_directory_exists(file_path: str) -> None:
    """Create directory if it doesn't exist."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def process_csv(file_path: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        # Detailed file investigation
        logging.info(f"Investigating file: {file_path}")
        
        # Check file existence and permissions
        if not os.path.exists(file_path):
            logging.error(f"File does not exist: {file_path}")
            return []
        
        # Check file size and readability
        file_size = os.path.getsize(file_path)
        logging.info(f"File size: {file_size} bytes")
        
        # Read raw file content for inspection
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            raw_content = f.read()
            logging.info(f"Raw file content:\n{raw_content[:500]}...")  # First 500 chars
        
        # Try multiple parsing strategies
        parsing_strategies = [
            lambda: pd.read_csv(file_path, delimiter='|'),
            lambda: pd.read_csv(file_path, delimiter=','),
            lambda: pd.read_csv(file_path, engine='python'),
            lambda: pd.read_csv(file_path, sep='\t')
        ]
        
        for strategy in parsing_strategies:
            try:
                df = strategy()
                if not df.empty and len(df.columns) > 1:
                    logging.info(f"Successfully parsed with strategy: {strategy}")
                    break
            except Exception as e:
                logging.debug(f"Strategy failed: {e}")
        else:
            logging.error("All parsing strategies failed")
            return []
        
        # Advanced delimiter detection
        delimiters = ['|', ',', ';', '\t']
        encodings = ['utf-8', 'latin-1', 'iso-8859-1']
        
        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    df = pd.read_csv(
                        file_path, 
                        encoding=encoding,
                        sep=delimiter,
                        engine='python',
                        quotechar='"',
                        skipinitialspace=True,
                        on_bad_lines='skip'
                    )
                    
                    if not df.empty and len(df.columns) > 1:
                        logging.info(f"Successfully parsed {file_path} with delimiter '{delimiter}' and encoding '{encoding}'")
                        break
                except Exception as inner_e:
                    logging.debug(f"Failed parsing with delimiter '{delimiter}' and encoding '{encoding}': {inner_e}")
            else:
                continue
            break
        else:
            logging.error(f"Could not parse {file_path} with any known delimiter or encoding")
            return []
        
        # Clean column names (remove extra whitespace, handle quotes)
        df.columns = [col.strip().strip('"') for col in df.columns]
        
        # Detect text columns more intelligently
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        logging.info(f"Processing {file_path}")
        logging.info(f"Text columns: {text_columns}")
        
        chunks = []
        
        # Process rows in chunks
        for i in range(0, len(df), config.get('CHUNK_SIZE', 5)):
            chunk_df = df.iloc[i:i+config.get('CHUNK_SIZE', 5)]
            
            # Combine text from selected columns
            chunk_texts = []
            for _, row in chunk_df.iterrows():
                row_text = " | ".join([
                    f"{col}: {str(row[col]).strip()}" 
                    for col in text_columns 
                    if pd.notna(row[col])
                ])
                chunk_texts.append(row_text)
            
            chunk_text = "\n".join(chunk_texts)
            
            # Create chunk metadata
            chunk = {
                "text": chunk_text,
                "metadata": {
                    "source": file_path,
                    "source_name": os.path.basename(file_path),
                    "document_type": "csv",
                    "chunk_number": (i // config.get('CHUNK_SIZE', 5)) + 1,
                    "rows": chunk_df.index.tolist(),
                    "text_columns": list(text_columns)
                }
            }
            
            chunks.append(chunk)
        
        logging.info(f"Created {len(chunks)} chunks from CSV")
        return chunks
    
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return []

def process_csvs(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Process all CSV files in a folder
    
    Args:
        config: Optional configuration dictionary
    
    Returns:
        Dictionary with processing results
    """
    # Merge provided config with default config
    processing_config = CSV_PROCESSING_CONFIG.copy()
    if config:
        processing_config.update(config)
    
    csv_folder = processing_config['CSV_FOLDER']
    output_file = processing_config['OUTPUT_FILE']
    
    # Ensure output directory exists
    ensure_directory_exists(output_file)
    
    # Find CSV files
    csv_files = [
        f for f in os.listdir(csv_folder) 
        if f.lower().endswith('.csv')
    ]
    
    if not csv_files:
        logging.warning(f"No CSV files found in {csv_folder}")
        return {'total_chunks': 0, 'chunks': []}
    
    logging.info(f"Found {len(csv_files)} CSV files")
    
    all_chunks = []
    
    # Process each CSV file
    for csv_file in tqdm(csv_files, desc="Processing CSVs"):
        csv_path = os.path.join(csv_folder, csv_file)
        chunks = process_csv(csv_path, processing_config)
        all_chunks.extend(chunks)
    
    # Save chunks to JSON
    if all_chunks:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved {len(all_chunks)} chunks to {output_file}")
        except Exception as e:
            logging.error(f"Error saving chunks: {e}")
    
    return {
        'total_chunks': len(all_chunks),
        'sample_chunk': all_chunks[0] if all_chunks else None,
        'chunks': all_chunks
    }

# Standalone testing
if __name__ == "__main__":
    result = process_csvs()
    
    print(f"Total Chunks: {result['total_chunks']}")
    
    if result['sample_chunk']:
        print("\nSample Chunk:")
        print(f"Source: {result['sample_chunk']['metadata']['source']}")
        print(f"Text preview: {result['sample_chunk']['text'][:150]}...")