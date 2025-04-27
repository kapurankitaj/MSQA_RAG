import os
import json
import numpy as np
import faiss
import pickle
import logging
import nltk
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Plus
from difflib import SequenceMatcher

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
except Exception as e:
    logging.warning(f"NLTK resource download error: {e}")
    # Fallback tokenizer in case NLTK fails
    def word_tokenize(text):
        return text.lower().split()

# ============= CONFIGURATION SETTINGS =============
CONFIG = {
    # Directories
    "directories": {
        "vector_db": "data/vector_db",
        "bm25_index": "data/bm25_index",
        "hybrid_search": "data/hybrid_search",
        "processed_data": "data/processed"
    },
    
    # Hybrid Search Parameters
    "hybrid_search": {
        "vector_weight": 0.6,   # Weight for vector search similarity
        "bm25_weight": 0.4,     # Weight for BM25 keyword score
        "top_k": 10,            # Number of results to retrieve
        "diversity_factor": 0.1 # Factor to penalize similar results (reduced from 0.2)
    },
    
    # Embedding Model
    "embedding": {
        "model_name": "all-MiniLM-L6-v2",
        "dimension": 384
    },
    
    # Logging configuration
    "logging": {
        "level": logging.INFO,
        "format": "%(asctime)s - %(levelname)s: %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S"
    }
}

# Configure logging
logging.basicConfig(
    level=CONFIG["logging"]["level"],
    format=CONFIG["logging"]["format"],
    datefmt=CONFIG["logging"]["date_format"]
)


class HybridSearchSystem:
    def __init__(self, config=CONFIG):
        """
        Initialize Hybrid Search System with vector and keyword search components
        
        Args:
            config (dict): Configuration dictionary with system parameters
        """
        # Store configuration
        self.config = config
        
        # Extract config values
        self.vector_db_dir = config["directories"]["vector_db"]
        self.bm25_dir = config["directories"]["bm25_index"]
        self.hybrid_search_dir = config["directories"]["hybrid_search"]
        self.vector_weight = config["hybrid_search"]["vector_weight"]
        self.bm25_weight = config["hybrid_search"]["bm25_weight"]
        self.top_k = config["hybrid_search"]["top_k"]
        self.diversity_factor = config["hybrid_search"]["diversity_factor"]
        
        # Ensure directories exist
        for directory in config["directories"].values():
            os.makedirs(directory, exist_ok=True)
        
        # Load vector search components
        self.load_vector_index()
        
        # Load BM25 components
        self.load_bm25_index()
        
        # Embedding model
        self.embedding_model = None
        self.load_embedding_model()
    
    def load_embedding_model(self):
        """Load the sentence transformer embedding model"""
        try:
            embedding_model_name = self.config["embedding"]["model_name"]
            self.embedding_model = SentenceTransformer(embedding_model_name)
            logging.info(f"Loaded embedding model: {embedding_model_name}")
        except Exception as e:
            logging.error(f"Error loading embedding model: {e}")
            raise
    
    def load_vector_index(self):
        """Load FAISS vector index and metadata"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.vector_db_dir, exist_ok=True)
            
            # Load FAISS index
            index_path = os.path.join(self.vector_db_dir, "faiss_index.index")
            if os.path.exists(index_path):
                self.vector_index = faiss.read_index(index_path)
                logging.info(f"Vector index loaded successfully from {index_path}")
            else:
                logging.warning(f"Vector index file not found at {index_path}")
                # Create empty index with correct dimension
                dimension = self.config["embedding"]["dimension"]
                self.vector_index = faiss.IndexFlatL2(dimension)
                logging.info(f"Created empty vector index with dimension {dimension}")
            
            # Try different metadata filenames (handle different naming conventions)
            potential_metadata_paths = [
                os.path.join(self.vector_db_dir, "chunk_metadata.pkl"),
                os.path.join(self.vector_db_dir, "metadata.pkl"),
                os.path.join(self.vector_db_dir, "vector_metadata.pkl")
            ]
            
            metadata_loaded = False
            for metadata_path in potential_metadata_paths:
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'rb') as f:
                            self.vector_metadata = pickle.load(f)
                        logging.info(f"Vector metadata loaded successfully from {metadata_path}")
                        metadata_loaded = True
                        break
                    except Exception as e:
                        logging.error(f"Error loading metadata from {metadata_path}: {e}")
            
            if not metadata_loaded:
                logging.warning(f"Vector metadata files not found in {self.vector_db_dir}")
                
                # If metadata is missing but index exists, create simple metadata
                if self.vector_index and self.vector_index.ntotal > 0:
                    logging.info(f"Creating basic metadata for {self.vector_index.ntotal} vectors")
                    self.vector_metadata = [{"id": i} for i in range(self.vector_index.ntotal)]
                    
                    # Save this basic metadata
                    metadata_save_path = os.path.join(self.vector_db_dir, "chunk_metadata.pkl")
                    try:
                        with open(metadata_save_path, 'wb') as f:
                            pickle.dump(self.vector_metadata, f)
                        logging.info(f"Basic metadata saved to {metadata_save_path}")
                    except Exception as e:
                        logging.error(f"Error saving basic metadata: {e}")
                else:
                    self.vector_metadata = []
                
        except Exception as e:
            logging.error(f"Error loading vector index: {e}")
            self.vector_index = None
            self.vector_metadata = []
    
    def load_bm25_index(self):
        """Load BM25 index and associated data"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.bm25_dir, exist_ok=True)
            
            # Set default empty values
            self.bm25 = None
            self.bm25_corpus = []
            self.bm25_metadata = []
            
            # Check and load BM25 index
            bm25_index_path = os.path.join(self.bm25_dir, "bm25_index.pkl")
            if os.path.exists(bm25_index_path):
                try:
                    with open(bm25_index_path, 'rb') as f:
                        # Check the structure of the saved data
                        bm25_data = pickle.load(f)
                        
                        # Handle different file structures that might exist
                        if isinstance(bm25_data, tuple) and len(bm25_data) == 3:
                            # Format: (bm25, corpus, metadata)
                            self.bm25, self.bm25_corpus, self.bm25_metadata = bm25_data
                            logging.info("Loaded BM25 index with metadata from tuple format")
                        elif isinstance(bm25_data, dict) and "bm25" in bm25_data:
                            # Dictionary format 
                            self.bm25 = bm25_data.get("bm25")
                            self.bm25_corpus = bm25_data.get("corpus", [])
                            self.bm25_metadata = bm25_data.get("metadata", [])
                            logging.info("Loaded BM25 index with metadata from dictionary format")
                        elif hasattr(bm25_data, "get_scores"):
                            # Just the BM25 object
                            self.bm25 = bm25_data
                            logging.info("Loaded BM25 index object, attempting to load corpus and metadata separately")
                            
                            # Try to load corpus and metadata from separate files
                            try:
                                corpus_path = os.path.join(self.bm25_dir, "bm25_corpus.pkl")
                                metadata_path = os.path.join(self.bm25_dir, "bm25_metadata.pkl")
                                
                                if os.path.exists(corpus_path):
                                    with open(corpus_path, 'rb') as fc:
                                        self.bm25_corpus = pickle.load(fc)
                                    logging.info(f"Loaded BM25 corpus from {corpus_path}")
                                
                                if os.path.exists(metadata_path):
                                    with open(metadata_path, 'rb') as fm:
                                        self.bm25_metadata = pickle.load(fm)
                                    logging.info(f"Loaded BM25 metadata from {metadata_path}")
                                    
                            except Exception as corpus_err:
                                logging.warning(f"BM25 corpus/metadata loading error: {corpus_err}")
                        else:
                            logging.warning(f"Unknown BM25 index format in {bm25_index_path}")
                            
                    # Verify the loaded data
                    if self.bm25 is not None:
                        logging.info(f"BM25 index loaded successfully from {bm25_index_path}")
                        if hasattr(self.bm25, "get_scores"):
                            logging.info("BM25 index has get_scores method")
                        else:
                            logging.warning("BM25 index does not have get_scores method")
                    
                    # Create metadata if missing but corpus exists
                    if not self.bm25_metadata and self.bm25_corpus:
                        logging.warning("BM25 metadata missing but corpus exists, creating basic metadata")
                        self.bm25_metadata = [{"id": i} for i in range(len(self.bm25_corpus))]
                        
                except Exception as pickle_err:
                    logging.error(f"Error unpickling BM25 index: {pickle_err}")
            else:
                logging.warning(f"BM25 index file not found at {bm25_index_path}")
                
        except Exception as e:
            logging.error(f"Error loading BM25 index: {e}")
            self.bm25 = None
            self.bm25_corpus = []
            self.bm25_metadata = []
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for the query"""
        if not self.embedding_model:
            self.load_embedding_model()
            
        return self.embedding_model.encode([query])[0]
    
    def vector_search(self, query_embedding: np.ndarray) -> List[Tuple[float, Dict]]:
        """Perform vector similarity search"""
        if self.vector_index is None:
            logging.warning("Vector index not loaded, returning empty results")
            return []
        
        try:
            # Normalize embedding
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search the index
            distances, indices = self.vector_index.search(query_embedding, self.top_k)
            
            # Check if we got any valid results
            if len(indices[0]) == 0:
                logging.warning("No vector search results found")
                return []
                
            # Convert to list of (distance, metadata)
            vector_results = []
            for i in range(len(indices[0])):
                if indices[0][i] < len(self.vector_metadata):
                    vector_results.append(
                        (float(distances[0][i]), self.vector_metadata[indices[0][i]])
                    )
                    
            logging.info(f"Vector search returned {len(vector_results)} results")
            return vector_results
            
        except Exception as e:
            logging.error(f"Error in vector search: {e}")
            return []
    
    def bm25_search(self, query: str) -> List[Tuple[float, Dict]]:
        """Perform BM25 keyword search"""
        if self.bm25 is None:
            logging.warning("BM25 index not loaded, returning empty results")
            return []
        
        try:
            # Simple tokenization with fallback if NLTK fails
            try:
                stop_words = set(stopwords.words('english'))
                query_tokens = [
                    token.lower() for token in word_tokenize(query) 
                    if token.lower() not in stop_words
                ]
            except Exception:
                # Fallback to simple tokenization
                query_tokens = [
                    token.lower() for token in query.split() 
                    if len(token) > 2  # Simple stopword filtering
                ]
            
            # Get BM25 scores
            doc_scores = self.bm25.get_scores(query_tokens)
            
            # Get top results
            top_indices = np.argsort(doc_scores)[::-1][:self.top_k]
            
            # Convert to list of (score, metadata)
            bm25_results = []
            for idx in top_indices:
                if doc_scores[idx] > 0 and idx < len(self.bm25_metadata):
                    bm25_results.append(
                        (float(doc_scores[idx]), self.bm25_metadata[idx])
                    )
            
            logging.info(f"BM25 search returned {len(bm25_results)} results")
            return bm25_results
            
        except Exception as e:
            logging.error(f"Error in BM25 search: {e}", exc_info=True)
            return []
    
    def compute_diversity_penalty(self, results: List[Tuple[float, Dict]]) -> List[Tuple[float, Dict]]:
        """
        Apply diversity penalty to reduce similarity between results
        
        Uses a simple approach to penalize results with too much overlap
        """
        if not results:
            return results
        
        try:
            # Sort results by score
            sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
            
            # Always include the top result
            final_results = [sorted_results[0]]
            
            # For each remaining result, check if it's too similar to any existing result
            for score, metadata in sorted_results[1:]:
                # Skip if no text available
                text = metadata.get('text', '')
                if not text:
                    final_results.append((score, metadata))
                    continue
                    
                # Compare with existing results
                too_similar = False
                for _, existing_metadata in final_results:
                    existing_text = existing_metadata.get('text', '')
                    if not existing_text:
                        continue
                        
                    # Calculate similarity
                    similarity = SequenceMatcher(None, text[:100], existing_text[:100]).ratio()
                    if similarity > self.diversity_factor:
                        too_similar = True
                        break
                
                # Add if not too similar
                if not too_similar:
                    final_results.append((score, metadata))
                    
                # Limit to top_k results
                if len(final_results) >= self.top_k:
                    break
            
            return final_results
        except Exception as e:
            logging.error(f"Error in diversity penalty calculation: {e}", exc_info=True)
            # If there's an error, just return the top K results without diversity filtering
            return sorted(results, key=lambda x: x[0], reverse=True)[:self.top_k]
    
    def hybrid_search(self, query: str) -> List[Dict]:
        """
        Perform hybrid search combining vector and keyword search

        1. Generate query embedding
        2. Perform vector and keyword searches
        3. Combine and re-rank results
        4. Apply diversity penalty
        """
        logging.info(f"Performing hybrid search for query: '{query}'")
        
        if not query.strip():
            logging.warning("Empty query received")
            return []
            
        try:
            # Generate query embedding
            query_embedding = self.generate_query_embedding(query)
            
            # Perform vector search
            vector_results = self.vector_search(query_embedding)
            
            # Perform BM25 search
            bm25_results = self.bm25_search(query)
            
            if not vector_results and not bm25_results:
                logging.warning("No results from either vector or BM25 search")
                return []
            
            # Normalize and combine results
            combined_results = []
            result_map = {}
            
            # Process vector results
            for vector_score, vector_metadata in vector_results:
                # Create a unique key for each metadata entry
                key = None
                if isinstance(vector_metadata, dict):
                    # Try to create a key from metadata fields
                    if 'id' in vector_metadata:
                        key = str(vector_metadata['id'])
                    elif 'text' in vector_metadata:
                        # Use first 100 chars of text as a key
                        key = vector_metadata['text'][:100]
                
                if key is None:
                    # Fallback to JSON representation
                    key = json.dumps(vector_metadata)
                
                result_map[key] = {
                    'metadata': vector_metadata,
                    'vector_score': vector_score,
                    'bm25_score': 0,
                    'hybrid_score': vector_score * self.vector_weight
                }
            
            # Process and combine BM25 results
            for bm25_score, bm25_metadata in bm25_results:
                # Create a unique key for each metadata entry
                key = None
                if isinstance(bm25_metadata, dict):
                    # Try to create a key from metadata fields
                    if 'id' in bm25_metadata:
                        key = str(bm25_metadata['id'])
                    elif 'text' in bm25_metadata:
                        # Use first 100 chars of text as a key
                        key = bm25_metadata['text'][:100]
                
                if key is None:
                    # Fallback to JSON representation
                    key = json.dumps(bm25_metadata)
                
                if key in result_map:
                    # Merge results if already exists from vector search
                    result_map[key]['bm25_score'] = bm25_score
                    result_map[key]['hybrid_score'] += bm25_score * self.bm25_weight
                else:
                    # New result from BM25
                    result_map[key] = {
                        'metadata': bm25_metadata,
                        'vector_score': 0,
                        'bm25_score': bm25_score,
                        'hybrid_score': bm25_score * self.bm25_weight
                    }
            
            # Convert to sorted list of results
            combined_results = sorted(
                result_map.values(), 
                key=lambda x: x['hybrid_score'], 
                reverse=True
            )[:self.top_k]
            
            logging.info(f"Combined search returned {len(combined_results)} results before diversity filtering")
            
            # Apply diversity penalty
            diverse_results = self.compute_diversity_penalty(
                [(result['hybrid_score'], result['metadata']) for result in combined_results]
            )
            
            # Create final results with original combined_results context
            final_results = []
            for score, metadata in diverse_results:
                # Find the corresponding result in combined_results
                for r in combined_results:
                    if r['metadata'] == metadata:
                        final_results.append({
                            'score': score,
                            'metadata': metadata,
                            'sources': [
                                f"Vector Score: {r['vector_score']:.4f}",
                                f"BM25 Score: {r['bm25_score']:.4f}"
                            ]
                        })
                        break
            
            logging.info(f"Final hybrid search returned {len(final_results)} results after diversity filtering")
            return final_results
            
        except Exception as e:
            logging.error(f"Error in hybrid search: {e}", exc_info=True)
            return []
    
    def explain_search_results(self, results: List[Dict]):
        """
        Provide detailed explanation of search results
        """
        if not results:
            print("\nNo results found. Please build your indexes first.")
            return
            
        print("\n" + "=" * 50)
        print("HYBRID SEARCH RESULTS EXPLANATION")
        print("=" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Hybrid Score: {result['score']:.4f}")
            
            # Print sources and scores
            for source in result.get('sources', []):
                print(f"  {source}")
            
            # Print metadata details
            print("Metadata:")
            metadata = result['metadata']
            for key, value in metadata.items():
                if key not in ['text', 'tokens']:  # Avoid printing long text
                    print(f"  {key}: {value}")
            
            # Print a snippet of the text if available
            if 'text' in metadata and metadata['text']:
                text = metadata['text']
                snippet = text[:200] + '...' if len(text) > 200 else text
                print(f"Text Snippet: {snippet}")
                
            print("-" * 50)


def build_indexes(config=CONFIG):
    """
    Build vector and BM25 indexes if they don't exist
    """
    import sys
    import importlib.util
    
    logging.info("Checking and building necessary indexes...")
    
    # Check if processed data exists
    processed_dir = config["directories"]["processed_data"]
    if not os.path.exists(processed_dir) or not os.listdir(processed_dir):
        logging.warning(f"No processed data found in {processed_dir}")
        logging.info("You need to process your source documents first")
        print(f"\nWARNING: No processed data found in {processed_dir}")
        print("You need to process your source documents first. Try running:")
        print("  python modules/csv_processing.py")
        print("  python modules/html_processing.py")
        print("  python modules/pdf_processing.py")
        print("  python modules/url_processing.py")
        return False
    
    success = True
    
    # Check if vector index exists
    vector_db_dir = config["directories"]["vector_db"]
    vector_index_path = os.path.join(vector_db_dir, "faiss_index.index")
    vector_metadata_path = os.path.join(vector_db_dir, "chunk_metadata.pkl")
    
    if not os.path.exists(vector_index_path) or not os.path.exists(vector_metadata_path):
        logging.info("Vector index or metadata not found. Building vector index...")
        try:
            # Try direct import of the module file
            vector_db_path = os.path.join(os.path.dirname(__file__), "vector_database.py")
            if os.path.exists(vector_db_path):
                spec = importlib.util.spec_from_file_location("vector_database", vector_db_path)
                vector_db_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(vector_db_module)
                
                # Call process_vector_database function
                result = vector_db_module.process_vector_database(config)
                if result:
                    logging.info(f"Vector index built successfully: {result}")
                else:
                    logging.error("Failed to build vector index")
                    success = False
            else:
                logging.error(f"Vector database module not found at {vector_db_path}")
                success = False
        except Exception as e:
            logging.error(f"Error building vector index: {e}")
            success = False
    
    # Check if BM25 index exists
    bm25_dir = config["directories"]["bm25_index"]
    bm25_index_path = os.path.join(bm25_dir, "bm25_index.pkl")
    
    if not os.path.exists(bm25_index_path):
        logging.info("BM25 index not found. Building BM25 index...")
        try:
            # Try direct import of the module file
            bm25_path = os.path.join(os.path.dirname(__file__), "bm25_search.py")
            if os.path.exists(bm25_path):
                spec = importlib.util.spec_from_file_location("bm25_search", bm25_path)
                bm25_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(bm25_module)
                
                # Call process_bm25 function
                result = bm25_module.process_bm25(config)
                if result:
                    logging.info(f"BM25 index built successfully: {result}")
                else:
                    logging.error("Failed to build BM25 index")
                    success = False
            else:
                logging.error(f"BM25 search module not found at {bm25_path}")
                success = False
        except Exception as e:
            logging.error(f"Error building BM25 index: {e}")
            success = False
    
    return success


def main():
    """
    Demonstrate hybrid search system functionality
    """
    # Display intro
    print("\n" + "=" * 80)
    print("HYBRID SEARCH SYSTEM")
    print("=" * 80)
    print("This module combines vector and keyword search for improved retrieval.")
    
    # Try to build indexes if they don't exist
    indexes_built = build_indexes()
    
    # Initialize Hybrid Search System
    hybrid_search = HybridSearchSystem()
    
    # Check if indexes are empty
    if (not hybrid_search.vector_metadata or hybrid_search.vector_index.ntotal == 0 or 
        not hybrid_search.bm25_metadata or hybrid_search.bm25 is None):
        print("\nWARNING: One or more indexes are empty or not loaded.")
        print("Follow these steps to prepare your search system:")
        print("1. Add your documents to the Files folder")
        print("2. Process your documents:")
        print("   python modules/csv_processing.py")
        print("   python modules/html_processing.py")
        print("   python modules/pdf_processing.py")
        print("   python modules/url_processing.py")
        print("3. Build your search indexes:")
        print("   python modules/vector_database.py")
        print("   python modules/bm25_search.py")
        print("\nWould you like to run the demonstration with empty indexes? (y/n)")
        
        user_input = input().strip().lower()
        if user_input != 'y' and user_input != 'yes':
            print("Exiting demonstration. Please build your indexes first.")
            return
        
        print("\nContinuing with demonstration using empty indexes...\n")
    
    # Example queries
    queries = [
        "What is retrieval augmented generation?",
        "How does vector search work?",
        "Machine learning techniques"
    ]
    
    # Perform hybrid search for each query
    for query in queries:
        print(f"\n\nQuery: {query}")
        print("=" * 50)
        
        # Perform hybrid search
        results = hybrid_search.hybrid_search(query)
        
        # Explain results
        hybrid_search.explain_search_results(results)


# Ensure proper script execution
if __name__ == "__main__":
    main()