"""
Integration of advanced features and optimization with the existing RAG system.
This script connects the new modules with the core RAG pipeline.
"""

import os
import yaml
import logging
import json
import time
import traceback
from typing import Dict, List, Optional, Any

# Import new modules
from advanced_features import AdvancedFeaturesManager
from optimization import OptimizationManager, PerformanceProfiler, CacheManager, BatchProcessor

# Setup logging
logging.basicConfig(
    filename='system_integration.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('system_integration')

class EnhancedRAGPipeline:
    """
    Enhanced RAG system that integrates advanced features and optimizations
    with the existing RAG pipeline.
    """
    
    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize the enhanced RAG system.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize base RAG pipeline
        from pipeline_integration import RAGPipeline
        self.base_pipeline = RAGPipeline(config_path)
        self.base_pipeline.initialize_pipeline()
        
        # Initialize LLM system for advanced features
        self.llm_system = self._get_llm_system()
        
        # Initialize new components
        self.advanced_features = AdvancedFeaturesManager(self.llm_system)
        
        optimization_config = self.config.get('optimization', {})
        self.optimization = OptimizationManager(
            max_cache_size_mb=optimization_config.get('max_cache_size_mb', 1000),
            batch_size=optimization_config.get('batch_size', 16),
            max_workers=optimization_config.get('max_workers', 4)
        )
        
        # Apply optimizations to existing components
        self._optimize_components()
        
        logger.info("EnhancedRAGPipeline initialized")
    
    def _get_llm_system(self):
        """Get LLM system if available in modules"""
        if 'llm_integration' in self.base_pipeline.modules:
            llm_module = self.base_pipeline.modules['llm_integration']
            if hasattr(llm_module, 'LLMIntegrationSystem'):
                try:
                    return llm_module.LLMIntegrationSystem()
                except Exception as e:
                    logger.error(f"Failed to initialize LLM system: {str(e)}")
        
        # Return a simple mock LLM service if no real one is available
        return type('MockLLMService', (), {
            'generate_text': lambda system_prompt, prompt, max_tokens: f"Response to: {prompt}"
        })
        
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict: Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            
            # Add default optimization and advanced features sections if missing
            if 'optimization' not in config:
                config['optimization'] = {
                    'max_cache_size_mb': 1000,
                    'batch_size': 16,
                    'max_workers': 4,
                    'enable_profiling': True,
                    'enable_caching': True,
                    'enable_memory_optimization': True,
                    'enable_batch_processing': True
                }
                
            if 'advanced_features' not in config:
                config['advanced_features'] = {
                    'enable_query_rewriting': True,
                    'enable_source_reliability': True,
                    'enable_personalization': True,
                    'enable_conversation_context': True
                }
                
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            logger.info("Using default configuration")
            
            # Return default configuration
            return {
                'data_sources': {
                    'pdf': {'enabled': True, 'path': 'data/pdfs'},
                    'html': {'enabled': True, 'path': 'data/html'},
                    'csv': {'enabled': True, 'path': 'data/csv'}
                },
                'optimization': {
                    'max_cache_size_mb': 1000,
                    'batch_size': 16,
                    'max_workers': 4,
                    'enable_profiling': True,
                    'enable_caching': True,
                    'enable_memory_optimization': True,
                    'enable_batch_processing': True
                },
                'advanced_features': {
                    'enable_query_rewriting': True,
                    'enable_source_reliability': True,
                    'enable_personalization': True,
                    'enable_conversation_context': True
                }
            }
            
    def _optimize_components(self):
        """Apply optimizations to existing pipeline components"""
        try:
            # Optimize the search_documents method in the base pipeline
            if hasattr(self.base_pipeline, 'search_documents'):
                original_search = self.base_pipeline.search_documents
                
                # Create an optimized version that uses the profiler
                def optimized_search(query, top_k=5):
                    if self.optimization.config['enable_profiling']:
                        self.optimization.profiler.start_profiling(f"search_{query[:20]}")
                    
                    try:
                        result = original_search(query, top_k)
                        return result
                    finally:
                        if self.optimization.config['enable_profiling']:
                            self.optimization.profiler.end_profiling()
                
                # Replace the original method with the optimized one
                self.base_pipeline.search_documents = optimized_search
                logger.info("Applied optimizations to search_documents method")
                
            else:
                logger.warning("Could not find search_documents method to optimize")
                
        except Exception as e:
            logger.error(f"Failed to optimize components: {str(e)}")
            
    def enhanced_process_query(self, query: str, user_id: str = 'default_user', 
                              conversation_context: Optional[List[Dict]] = None) -> Dict:
        """
        Process a query using enhanced features and optimizations.
        
        Args:
            query: User query text
            user_id: User identifier for personalization
            conversation_context: Optional conversation context
            
        Returns:
            Dict: Query response with enhanced features
        """
        start_time = time.time()
        
        # Start profiling if enabled
        if self.optimization.config['enable_profiling']:
            self.optimization.profiler.start_profiling(f"query_{user_id}")
            
        try:
            logger.info(f"Processing query: {query}")
            
            # Step 1: Apply advanced features to enhance the query
            enhanced_query = self.advanced_features.enhance_query(user_id, query)
            logger.info(f"Enhanced query: {enhanced_query}")
            
            # Step 2: Use the base pipeline to get search results and response
            # We're adapting the existing process_query method from RAGPipeline
            base_response = self.base_pipeline.process_query(enhanced_query, conversation_context)
            
            # Get the retrieved documents from the response metadata
            if 'metadata' in base_response and 'sources' in base_response['metadata']:
                sources = base_response['metadata']['sources']
                # Convert sources to document objects for enhance_results
                documents = [{'id': src, 'metadata': {'source': src}} for src in sources]
                
                # Step 3: Apply source reliability scoring and personalization
                if documents:
                    enhanced_docs = self.advanced_features.enhance_results(user_id, documents)
                    
                    # Update response with enhanced document info
                    base_response['metadata']['enhanced_documents'] = [
                        {'id': doc.get('id'), 'reliability_score': doc.get('reliability_score', 0)}
                        for doc in enhanced_docs if 'reliability_score' in doc
                    ]
            
            # Step 4: Add enhancement info to the response
            base_response['metadata']['query_enhancement'] = {
                'original_query': query,
                'enhanced_query': enhanced_query
            }
            
            # Step 5: Store interaction for conversation context
            self.advanced_features.store_interaction(
                user_id, 
                query, 
                base_response['response'], 
                documents if 'documents' in locals() else []
            )
            
            # Add execution time
            base_response['metadata']['execution_time'] = time.time() - start_time
            
            return base_response
            
        except Exception as e:
            logger.error(f"Error in enhanced query processing: {str(e)}")
            logger.error(traceback.format_exc())
            
            return {
                "response": "An error occurred while processing your query with enhanced features.",
                "metadata": {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "execution_time": time.time() - start_time
                }
            }
        finally:
            # End profiling if enabled
            if self.optimization.config['enable_profiling']:
                profile_results = self.optimization.profiler.end_profiling()
                logger.info(f"Query processing completed in {profile_results.get('total_time', 0):.2f} seconds")
    
    def batch_process_queries(self, queries: List[str], user_id: str = 'default_user') -> List[Dict]:
        """
        Process multiple queries efficiently using batch processing.
        
        Args:
            queries: List of query strings
            user_id: User identifier for personalization
            
        Returns:
            List[Dict]: List of query responses
        """
        if not self.optimization.config['enable_batch_processing']:
            # Process sequentially if batch processing is disabled
            return [self.enhanced_process_query(q, user_id) for q in queries]
        
        # Define batch processor function
        def process_query_batch(query_batch):
            return [self.enhanced_process_query(q, user_id) for q in query_batch]
        
        # Use batch processor
        return self.optimization.batch_processor.process_batches(
            queries, process_query_batch, use_threading=True
        )
    
    def get_system_stats(self) -> Dict:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dict: System statistics
        """
        stats = {
            'optimization': self.optimization.get_optimization_stats(),
            'advanced_features': {
                'personalization': {
                    'user_count': len(self.advanced_features.personalization.user_profiles)
                },
                'conversation': {
                    'conversation_count': len(self.advanced_features.conversation.conversations)
                }
            }
        }
        
        return stats


def main():
    """Demo of the enhanced RAG pipeline"""
    # Initialize the enhanced pipeline
    pipeline = EnhancedRAGPipeline()
    
    # Example query
    query = "What is retrieval augmented generation?"
    
    # Process with enhanced features
    response = pipeline.enhanced_process_query(query, user_id="example_user")
    
    # Print results
    print("\nQuery:", query)
    print("\nEnhanced Response:", response['response'])
    print("\nMetadata:", json.dumps(response['metadata'], indent=2))
    
    # Show system stats
    stats = pipeline.get_system_stats()
    print("\nSystem Stats:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()