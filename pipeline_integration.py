import os
import yaml
import logging
import functools
import traceback
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, Any, List, Optional
import time

try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    logging.warning("SpellChecker not available. Install with: pip install pyspellchecker")

# Spelling correction
try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    logging.warning("SpellChecker not available. Install with: pip install pyspellchecker")

# ============= CONFIGURATION SETTINGS =============
CONFIG_FILE = "config/rag_config.yaml"
DEFAULT_DATA_SOURCES = {
    'pdf': {'enabled': True, 'path': 'data/pdfs'},
    'html': {'enabled': True, 'path': 'data/html'},
    'csv': {'enabled': True, 'path': 'data/csv'},
    'sql': {'enabled': True, 'path': 'data/sql'},
    'url': {'enabled': True, 'path': 'data/urls'}
}
MAX_WORKERS = 4
CACHE_SIZE = 100
LOG_LEVEL = 'INFO'
LOG_FILE = 'logs/pipeline_integration.log'

class PipelineConfig:
    def __init__(self, config_path='config/config.yaml'):
        self.config = self.load_config(config_path)
        self.setup_logging()

    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logging.warning(f"Config file {config_path} not found. Using default configuration.")
            return self.create_default_config()

    def create_default_config(self):
        return {
            'data_sources': {
                'pdf': {'enabled': True, 'path': 'data/pdfs'},
                'html': {'enabled': True, 'path': 'data/html'},
                'csv': {'enabled': True, 'path': 'data/csv'}
            },
            'retrieval': {
                'vector_db': {'top_k': 5},
                'bm25': {'top_k': 3}
            },
            'generation': {
                'max_tokens': 1000,
                'temperature': 0.7
            },
            'logging': {
                'level': 'INFO',
                'file': 'pipeline.log'
            },
            'max_workers': 4
        }

    def setup_logging(self):
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        log_file = self.config.get('logging', {}).get('file', 'pipeline.log')
        
        # Make sure log_file has a valid directory
        if not os.path.dirname(log_file):
            log_file = os.path.join('logs', log_file)
        
        # Ensure logs directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

class RAGPipeline:
    def __init__(self, config_path='config/config.yaml'):
        self.config = PipelineConfig(config_path).config
        self.processor = PipelineProcessor(self.config)
        self.modules = self.processor.modules
        
        # Initialize components
        self.vector_db = None
        self.bm25_search = None
        self.llm_system = None
        self.citation_manager = None
        
        # Directory paths
        self.data_processed_dir = 'data/processed'
        self.vector_db_dir = 'data/vector_db'
        self.bm25_dir = 'data/bm25_index'
        
        # Ensure directories exist
        for directory in [self.data_processed_dir, self.vector_db_dir, self.bm25_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize spell checker if available
        self.spell_checker = None
        if SPELLCHECKER_AVAILABLE:
            self.spell_checker = SpellChecker()
            # Add domain-specific words to avoid correcting them
            self.spell_checker.word_frequency.load_words([
                'langchain', 'langgraph', 'vector', 'embedding', 'transformer', 
                'retrieval', 'augmented', 'generation', 'rag'
            ])
        
    def initialize_pipeline(self):
        """Initialize pipeline components"""
        try:
            # Initialize vector database
            if 'vector_database' in self.modules:
                vector_db_module = self.modules['vector_database']
                self.vector_db = getattr(vector_db_module, 'process_vector_database', None)
            
            # Initialize BM25 search
            if 'bm25_search' in self.modules:
                bm25_module = self.modules['bm25_search']
                self.bm25_search = getattr(bm25_module, 'process_bm25', None)
            
            # Check and build indices if needed
            self._check_and_build_indices()
            
            logging.info("Pipeline initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Pipeline initialization failed: {e}")
            logging.error(traceback.format_exc())
            return False
    
    def _check_and_build_indices(self):
        """Check if vector and BM25 indices exist and build them if needed"""
        # Check for processed data first
        processed_files = []
        for root, _, files in os.walk(self.data_processed_dir):
            processed_files.extend([f for f in files if f.endswith('.json')])
        
        if not processed_files:
            logging.warning(f"No processed data found in {self.data_processed_dir}. Please process documents first.")
            return False
        
        logging.info(f"Found {len(processed_files)} processed data files")
        
        # Check and build vector index
        vector_index_path = os.path.join(self.vector_db_dir, 'faiss_index.index')
        vector_metadata_path = os.path.join(self.vector_db_dir, 'chunk_metadata.pkl')
        
        if not os.path.exists(vector_index_path) or not os.path.exists(vector_metadata_path):
            logging.info("Building vector database index...")
            if self.vector_db:
                try:
                    self.vector_db()
                    logging.info("Vector index built successfully")
                except Exception as e:
                    logging.error(f"Failed to build vector index: {e}")
            else:
                logging.error("Vector database module not available")
        else:
            logging.info("Vector index already exists")
        
        # Check and build BM25 index
        bm25_index_path = os.path.join(self.bm25_dir, 'bm25_index.pkl')
        
        if not os.path.exists(bm25_index_path):
            logging.info("Building BM25 index...")
            if self.bm25_search:
                try:
                    self.bm25_search()
                    logging.info("BM25 index built successfully")
                except Exception as e:
                    logging.error(f"Failed to build BM25 index: {e}")
            else:
                logging.error("BM25 search module not available")
        else:
            logging.info("BM25 index already exists")
            
        return True
    
    def _correct_spelling(self, text: str) -> str:
        """Correct spelling mistakes in query"""
        if not self.spell_checker:
            return text
            
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Skip correction for special terms, URLs, etc.
            if len(word) < 4 or '/' in word or '@' in word or word.lower() in ['the', 'and', 'for']:
                corrected_words.append(word)
                continue
                
            # Check if misspelled
            if word.lower() in self.spell_checker.unknown([word.lower()]):
                corrected = self.spell_checker.correction(word.lower())
                if corrected and corrected != word.lower():
                    logging.info(f"Corrected '{word}' to '{corrected}'")
                    corrected_words.append(corrected)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search across different document types"""
        results = []
        
        # Use hybrid search if available and return early if successful
        if 'hybrid_search' in self.modules:
            try:
                hybrid_search_system = self.modules['hybrid_search'].HybridSearchSystem()
                hybrid_results = hybrid_search_system.hybrid_search(query)
                if hybrid_results:
                    logging.info(f"Found {len(hybrid_results)} results using hybrid search")
                    results = hybrid_results[:top_k]
                    # Skip other searches if we have results
                    if results:
                        pass  # Continue to text loading
                    else:
                        logging.warning("Hybrid search returned no results, falling back to individual searches")
                else:
                    logging.warning("Hybrid search returned no results, falling back to individual searches")
            except Exception as e:
                logging.error(f"Hybrid search failed: {e}")
                logging.info("Falling back to individual search methods")
        
        # Use ThreadPoolExecutor for parallel search if needed
        if not results:
            search_tasks = {}
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit vector search task
                if 'vector_database' in self.modules:
                    # Find vector search function
                    vector_search_func = None
                    if hasattr(self.modules['vector_database'], 'query_vector_database'):
                        vector_search_func = self.modules['vector_database'].query_vector_database
                    elif hasattr(self.modules['vector_database'], 'search_vector_database'):
                        vector_search_func = self.modules['vector_database'].search_vector_database
                    
                    if vector_search_func:
                        search_tasks['vector'] = executor.submit(vector_search_func, query, top_k=top_k)
                
                # Submit BM25 search task
                if 'bm25_search' in self.modules:
                    bm25_search_func = None
                    if hasattr(self.modules['bm25_search'], 'search_bm25'):
                        bm25_search_func = self.modules['bm25_search'].search_bm25
                    
                    if bm25_search_func:
                        search_tasks['bm25'] = executor.submit(bm25_search_func, query, top_k=top_k)
                
                # Collect results from completed tasks
                for name, future in search_tasks.items():
                    try:
                        task_results = future.result()
                        if task_results:
                            logging.info(f"{name} search found {len(task_results)} results")
                            results.extend(task_results)
                            # If we have enough results already, cancel remaining tasks
                            if len(results) >= top_k:
                                for n, f in search_tasks.items():
                                    if n != name and not f.done():
                                        f.cancel()
                                break
                        else:
                            logging.warning(f"{name} search returned no results")
                    except Exception as e:
                        logging.error(f"{name} search failed: {e}")
                        logging.error(traceback.format_exc())
        
        # If no results found, try to generate a mock response
        if not results:
            logging.warning("No search results found. Creating mock result.")
            results = [{
                'score': 0.5,
                'metadata': {'source': 'Mock Source', 'document_type': 'mock'},
                'text': f"No results found for query: {query}. This is a mock response."
            }]
        
        # Load text content for all results
        for result in results:
            if 'text' not in result:
                source_file = result['metadata'].get('source_file')
                if source_file and os.path.exists(source_file):
                    try:
                        with open(source_file, 'r', encoding='utf-8') as f:
                            all_chunks = json.load(f)
                            # Find matching chunk
                            for chunk in all_chunks:
                                chunk_meta = chunk.get('metadata', {})
                                if (chunk_meta.get('source') == result['metadata'].get('source') and
                                    chunk_meta.get('chunk_number') == result['metadata'].get('chunk_number')):
                                    result['text'] = chunk.get('text', '')
                                    logging.info(f"Loaded text content for {result['metadata'].get('source')}")
                                    break
                    except Exception as e:
                        logging.error(f"Error loading text from {source_file}: {e}")
                
                # If text still not found, add placeholder
                if 'text' not in result:
                    result['text'] = f"Document: {result['metadata'].get('title', 'Unknown')}"
        
        # Return top_k results, sorted by relevance
        return sorted(results, key=lambda x: x.get('score', 0), reverse=True)[:top_k]


    def process_query(self, query: str, conversation_context: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Process a query through the full RAG pipeline"""
        start_time = time.time()
        
        try:
            # Apply spelling correction
            original_query = query
            query = self._correct_spelling(query)
            if query != original_query:
                logging.info(f"Corrected query from '{original_query}' to '{query}'")
            
            # Attempt to route query using query_routing module if available
            if 'query_routing' in self.modules:
                try:
                    query_router = self.modules['query_routing'].QueryRouter()
                    routing_result = query_router.route_query(query, conversation_context)
                    logging.info(f"Query routed as type: {routing_result.get('query_info', {}).get('query_type', 'unknown')}")
                except Exception as e:
                    logging.error(f"Query routing failed: {e}")
                    routing_result = None
            else:
                routing_result = None
            
            # Search for relevant documents
            context_docs = self.search_documents(query)
            
            # Use LLM integration if available
            if 'llm_integration' in self.modules and hasattr(self.modules['llm_integration'], 'LLMIntegrationSystem'):
                try:
                    llm_system = self.modules['llm_integration'].LLMIntegrationSystem()
                    response_data = llm_system.generate_response(query, context_docs, conversation_context)
                    logging.info("Generated response using LLM integration")
                    
                    # Add citation if available
                    if 'citation_system' in self.modules and hasattr(self.modules['citation_system'], 'CitationManager'):
                        try:
                            citation_manager = self.modules['citation_system'].CitationManager()
                            citations = citation_manager.process_sources(context_docs)
                            formatted_response = citation_manager.format_response_with_citations(
                                response_data['response'], citations)
                            response_data['response'] = formatted_response['text']
                            response_data['metadata']['citations'] = citations
                            logging.info("Added citations to response")
                        except Exception as e:
                            logging.error(f"Citation processing failed: {e}")
                    
                    # Add spelling correction info if applied
                    if query != original_query:
                        if 'metadata' not in response_data:
                            response_data['metadata'] = {}
                        response_data['metadata']['corrected_query'] = {
                            'original': original_query,
                            'corrected': query
                        }
                    
                    return response_data
                except Exception as e:
                    logging.error(f"LLM integration failed: {e}")
                    logging.error(traceback.format_exc())
            
            # Fall back to a basic response
            response = {
                "response": f"Response to query: {query}\n\nFound {len(context_docs)} relevant documents.",
                "metadata": {
                    "search_results": len(context_docs),
                    "sources": [doc.get('metadata', {}).get('source', 'Unknown') for doc in context_docs],
                    "execution_time": time.time() - start_time
                }
            }
            
            # Add spelling correction info if applied
            if query != original_query:
                response['metadata']['corrected_query'] = {
                    'original': original_query,
                    'corrected': query
                }
            
            if context_docs:
                response["response"] += "\n\nRelevant information:\n"
                for i, doc in enumerate(context_docs[:3], 1):
                    response["response"] += f"\n{i}. {doc.get('text', 'No text available')[:200]}...\n"
            
            return response
        
        except Exception as e:
            logging.error(f"Query processing error: {e}")
            logging.error(traceback.format_exc())
            return {
                "response": "An error occurred while processing your query.",
                "metadata": {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "execution_time": time.time() - start_time
                }
            }

def handle_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            logging.error(traceback.format_exc())
            raise
    return wrapper

class PipelineCache:
    @staticmethod
    def generate_cache_key(*args, **kwargs):
        key_str = json.dumps(args) + json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    @classmethod
    @lru_cache(maxsize=100)
    def cached_process(cls, func, *args, **kwargs):
        return func(*args, **kwargs)

class PipelineProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_workers = config.get('max_workers', 4)
        self.modules = self._import_modules()

    def _import_modules(self):
        """Import and return available processing modules"""
        modules = {}
        module_names = [
            'pdf_processing', 'html_processing', 'csv_processing', 
            'sql_database', 'url_processing', 'vector_database',
            'bm25_search', 'hybrid_search', 'llm_integration',
            'citation_system', 'query_routing', 'ragas_metrics'
        ]
        
        for module_name in module_names:
            try:
                module = __import__(f'modules.{module_name}', fromlist=[''])
                modules[module_name] = module
                logging.info(f"Successfully imported {module_name}")
            except ImportError as e:
                logging.warning(f"Could not import {module_name}: {e}")
        
        return modules
            
def main():
    # Demonstrate RAGPipeline usage
    pipeline = RAGPipeline()
    pipeline.initialize_pipeline()
    
    # Example query
    query = "What is retrieval augmented generation?"
    response = pipeline.process_query(query)
    
    print("\nQuery:", query)
    print("\nResponse:", response['response'])
    print("\nMetadata:", json.dumps(response['metadata'], indent=2))

if __name__ == "__main__":
    main()