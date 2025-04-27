# modules/vector_database.py
import os
import json
import numpy as np
import faiss
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

# ============= CONFIGURATION SETTINGS =============
VECTOR_DB_CONFIG = {
    'PROCESSED_DATA_DIR': "data/processed",
    'VECTOR_DB_DIR': "data/vector_db",
    'EMBEDDINGS_CACHE_DIR': "data/vector_db/embeddings_cache",
    'EMBEDDING_MODEL': "all-MiniLM-L6-v2",
    'EMBEDDING_DIMENSION': 384,
    'INDEX_TYPE': "L2",
    'NORMALIZE_EMBEDDINGS': True,
    
    # Logging configuration
    'LOG_LEVEL': logging.INFO,
    'LOG_FORMAT': '%(asctime)s - %(levelname)s: %(message)s',
    'LOG_DATE_FORMAT': '%Y-%m-%d %H:%M:%S'
}

# Configure logging
logging.basicConfig(
    level=VECTOR_DB_CONFIG['LOG_LEVEL'], 
    format=VECTOR_DB_CONFIG['LOG_FORMAT'],
    datefmt=VECTOR_DB_CONFIG['LOG_DATE_FORMAT']
)

class VectorDatabase:
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize vector database with configuration."""
        self.config = VECTOR_DB_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # Ensure directories exist
        os.makedirs(self.config['VECTOR_DB_DIR'], exist_ok=True)
        os.makedirs(self.config['EMBEDDINGS_CACHE_DIR'], exist_ok=True)
        
        self.model = None
        self.index = None
        self.metadata = None
    
    def load_model(self):
        """Load embedding model."""
        try:
            self.model = SentenceTransformer(self.config['EMBEDDING_MODEL'])
            logging.info(f"Loaded embedding model: {self.config['EMBEDDING_MODEL']}")
        except Exception as e:
            logging.error(f"Error loading embedding model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            self.load_model()
        
        try:
            # Dramatically reduce texts for initial testing
            if len(texts) > 1000:
                import random
                texts = random.sample(texts, 1000)
                logging.warning(f"Testing with {len(texts)} sampled texts")
            
            # Batch processing with aggressive optimization
            batch_size = 512  # Even larger batch size
            all_embeddings = []
            
            from tqdm import tqdm
            
            for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts, 
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    batch_size=64,  # Increased internal batch size
                    device='cpu'
                )
                
                if self.config['NORMALIZE_EMBEDDINGS']:
                    faiss.normalize_L2(batch_embeddings)
                
                all_embeddings.append(batch_embeddings)
            
            embeddings_array = np.concatenate(all_embeddings, axis=0)
            
            logging.info(f"Generated {len(embeddings_array)} embeddings")
            return embeddings_array
        
        except Exception as e:
            logging.error(f"Embedding generation error: {e}")
            return np.zeros((len(texts), self.config['EMBEDDING_DIMENSION']))
    
    def create_index(self, embeddings: np.ndarray):
        """Create FAISS index from embeddings."""
        try:
            dimension = embeddings.shape[1]
            
            if self.config['INDEX_TYPE'] == "L2":
                self.index = faiss.IndexFlatL2(dimension)
            else:
                self.index = faiss.IndexFlatIP(dimension)
            
            self.index.add(embeddings)
            logging.info(f"Created index with {self.index.ntotal} vectors")
        except Exception as e:
            logging.error(f"Error creating FAISS index: {e}")
    
    def save_index(self, index_path: Optional[str] = None):
        """Save FAISS index to disk."""
        if not self.index:
            logging.error("No index to save")
            return
        
        index_path = index_path or os.path.join(
            self.config['VECTOR_DB_DIR'], 
            'faiss_index.index'
        )
        
        try:
            faiss.write_index(self.index, index_path)
            logging.info(f"Saved index to {index_path}")
        except Exception as e:
            logging.error(f"Error saving index: {e}")
    
    def load_processed_data(self, data_dir: Optional[str] = None):
        """Load processed data from JSON files."""
        data_dir = data_dir or self.config['PROCESSED_DATA_DIR']
        
        texts = []
        metadata = []
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(data_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        chunks = json.load(f)
                    
                    for chunk in chunks:
                        texts.append(chunk.get('text', ''))
                        metadata.append(chunk.get('metadata', {}))
                except Exception as e:
                    logging.error(f"Error loading {filename}: {e}")
        
        return texts, metadata

def process_vector_database(config: Dict[str, Any] = None):
    """Main function to process vector database."""
    try:
        db = VectorDatabase(config)
        
        # Load processed data
        texts, metadata = db.load_processed_data()
        
        # Logging to help diagnose any issues
        logging.info(f"Found {len(texts)} texts to process")
        
        if not texts:
            logging.error("No texts found for vector database")
            return None
        
        # Generate embeddings
        logging.info("Generating embeddings...")
        embeddings = db.generate_embeddings(texts)
        
        # Log embeddings details
        logging.info(f"Embeddings shape: {embeddings.shape}")
        
        # Create FAISS index
        logging.info("Creating FAISS index...")
        db.create_index(embeddings)
        
        # Save index
        logging.info("Saving index...")
        db.save_index()
        
        return {
            'total_chunks': len(texts),
            'embedding_model': db.config['EMBEDDING_MODEL'],
            'index_type': db.config['INDEX_TYPE']
        }
    
    except Exception as e:
        logging.error(f"Vector database processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# Standalone testing
if __name__ == "__main__":
    result = process_vector_database()
    print("Vector Database Processing Result:")
    print(result)