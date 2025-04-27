# modules/bm25_search.py
import os
import json
import pickle
import logging
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from rank_bm25 import BM25Okapi

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ============= CONFIGURATION SETTINGS =============
BM25_CONFIG = {
    'PROCESSED_DATA_DIR': "data/processed",
    'BM25_DIR': "data/bm25_index",
    'BM25_VARIANT': "BM25Okapi",
    'LOWERCASE': True,
    'REMOVE_STOPWORDS': True,
    
    # Logging configuration
    'LOG_LEVEL': logging.INFO,
    'LOG_FORMAT': '%(asctime)s - %(levelname)s: %(message)s',
    'LOG_DATE_FORMAT': '%Y-%m-%d %H:%M:%S'
}

# Configure logging
logging.basicConfig(
    level=BM25_CONFIG['LOG_LEVEL'], 
    format=BM25_CONFIG['LOG_FORMAT'],
    datefmt=BM25_CONFIG['LOG_DATE_FORMAT']
)

class BM25Processor:
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize BM25 processor with configuration."""
        self.config = BM25_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # Ensure output directory exists
        os.makedirs(self.config['BM25_DIR'], exist_ok=True)
        
        # Download NLTK resources if needed
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            logging.warning(f"NLTK resource download failed: {e}")
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess and tokenize text."""
        if not text or not isinstance(text, str):
            return []
        
        # Lowercase if configured
        if self.config['LOWERCASE']:
            text = text.lower()
        
        # Use simple tokenization instead of word_tokenize
        tokens = text.split()  # Simple splitting by whitespace
        
        # Remove stopwords if configured
        if self.config['REMOVE_STOPWORDS']:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
        
        return tokens
    
    def load_chunks(self, directory: str) -> Tuple[List[str], List[Dict]]:
        """Load text chunks from JSON files."""
        all_texts = []
        all_metadata = []
        
        json_files = [
            os.path.join(root, f) 
            for root, _, files in os.walk(directory) 
            for f in files if f.lower().endswith('.json')
        ]
        
        logging.info(f"Found {len(json_files)} JSON files")
        
        for json_file in tqdm(json_files, desc="Loading chunks"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                
                for chunk in chunks:
                    if chunk.get('text', '').strip():
                        all_texts.append(chunk['text'])
                        metadata = chunk.get('metadata', {})
                        metadata['source_file'] = json_file
                        all_metadata.append(metadata)
            
            except Exception as e:
                logging.error(f"Error loading {json_file}: {e}")
        
        logging.info(f"Loaded {len(all_texts)} chunks")
        return all_texts, all_metadata
    
    def create_bm25_index(self, texts: List[str]) -> Any:
        """Create BM25 index from texts."""
        tokenized_corpus = [self.preprocess_text(text) for text in texts]
        
        # Remove empty token lists
        tokenized_corpus = [tokens for tokens in tokenized_corpus if tokens]
        
        logging.info(f"Creating BM25 index with {len(tokenized_corpus)} documents")
        
        try:
            bm25 = BM25Okapi(tokenized_corpus)
            return bm25, tokenized_corpus
        except Exception as e:
            logging.error(f"BM25 index creation failed: {e}")
            return None, None

def process_bm25(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Process BM25 keyword search index."""
    processor = BM25Processor(config)
    
    # Load chunks
    texts, metadata = processor.load_chunks(
        processor.config['PROCESSED_DATA_DIR']
    )
    
    # Create BM25 index
    bm25, tokenized_corpus = processor.create_bm25_index(texts)
    
    if not bm25:
        logging.error("Failed to create BM25 index")
        return None
    
    # Save index (optional)
    index_path = os.path.join(
        processor.config['BM25_DIR'], 
        'bm25_index.pkl'
    )
    
    with open(index_path, 'wb') as f:
        pickle.dump((bm25, tokenized_corpus, metadata), f)
    
    return {
        'total_chunks': len(texts),
        'index_path': index_path
    }

# Standalone testing
if __name__ == "__main__":
    result = process_bm25()
    print(result)