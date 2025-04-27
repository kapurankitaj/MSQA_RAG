import os
import json
import time
import logging
import unittest
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# ============= CONFIGURATION SETTINGS =============
CONFIG = {
    # Test data paths
    "paths": {
        "test_data_dir": "data/test",
        "test_output_dir": "data/test/results",
        "test_log_file": "testing_framework.log"
    },
    
    # Sample data sizes
    "sample_sizes": {
        "num_pdf_test_files": 2,
        "num_html_test_files": 2,
        "num_csv_test_files": 2,
        "num_url_test_links": 2,
        "num_test_queries": 10
    },
    
    # Test parameters
    "test_settings": {
        "enable_performance_tests": True,
        "enable_integration_tests": True,
        "enable_component_tests": True,
        "performance_benchmark_iterations": 3,
        "test_timeout_seconds": 30
    },
    
    # Test reporting settings
    "reporting": {
        "generate_html_report": True,
        "save_test_results": True,
        "verbose_output": True
    }
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["paths"]["test_log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TestingFramework")

class RAGTestCase(unittest.TestCase):
    """Base class for RAG system test cases"""
    
    def setUp(self):
        """Set up test case with common utilities"""
        # Create test directories if they don't exist
        os.makedirs(CONFIG["paths"]["test_data_dir"], exist_ok=True)
        os.makedirs(CONFIG["paths"]["test_output_dir"], exist_ok=True)
        
        # Record start time for performance tracking
        self.start_time = time.time()
        logger.info(f"Starting test: {self._testMethodName}")
    
    def tearDown(self):
        """Clean up after test case execution"""
        # Calculate and record test execution time
        execution_time = time.time() - self.start_time
        logger.info(f"Test completed in {execution_time:.2f} seconds: {self._testMethodName}")
    
    def assertDocumentFormat(self, document: Dict[str, Any]):
        """Assert that a document follows the expected format"""
        self.assertIsInstance(document, dict)
        self.assertIn("text", document)
        self.assertIn("metadata", document)
        self.assertIsInstance(document["metadata"], dict)
    
    def assertValidEmbedding(self, embedding: np.ndarray, expected_dim: int = 384):
        """Assert that an embedding is valid"""
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.dtype, np.float32)
        self.assertEqual(embedding.shape, (expected_dim,))
    
    def assertValidSearchResults(self, results: List[Dict[str, Any]], min_results: int = 1):
        """Assert that search results have valid format"""
        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), min_results)
        for result in results:
            self.assertIn("score", result)
            self.assertIn("metadata", result)


class DataIngestionTests(RAGTestCase):
    """Tests for the data ingestion components"""
    
    def test_pdf_processing(self):
        """Test PDF processing pipeline"""
        try:
            # Import PDF processing module
            import sys
            sys.path.append(".")
            # This would be your actual import
            # from pdf_processing import process_pdf, chunk_text
            
            # Mock for testing purposes
            def process_pdf(file_path, chunk_size=1000, chunk_overlap=200):
                return [{"text": "Test content", "metadata": {"source": file_path, "page_number": 1}}]
            
            # Create test PDF or use sample
            test_pdf_path = os.path.join(CONFIG["paths"]["test_data_dir"], "test_document.pdf")
            if not os.path.exists(test_pdf_path):
                logger.warning(f"Test PDF not found at {test_pdf_path}. Test will use mock data.")
            
            # Process test PDF
            chunks = process_pdf(test_pdf_path)
            
            # Validate results
            self.assertIsInstance(chunks, list)
            if chunks:
                self.assertDocumentFormat(chunks[0])
                
        except Exception as e:
            logger.error(f"Error in PDF processing test: {e}")
            self.fail(f"PDF processing test failed: {e}")
    
    def test_html_processing(self):
        """Test HTML processing pipeline"""
        try:
            # Mock for testing purposes
            def process_html_simplified(file_path, chunk_size=1000, chunk_overlap=200):
                return [{"text": "Test HTML content", "metadata": {"source": file_path}}]
            
            # Create test HTML or use sample
            test_html_path = os.path.join(CONFIG["paths"]["test_data_dir"], "test_page.html")
            if not os.path.exists(test_html_path):
                logger.warning(f"Test HTML not found at {test_html_path}. Test will use mock data.")
            
            # Process test HTML
            chunks = process_html_simplified(test_html_path)
            
            # Validate results
            self.assertIsInstance(chunks, list)
            if chunks:
                self.assertDocumentFormat(chunks[0])
                
        except Exception as e:
            logger.error(f"Error in HTML processing test: {e}")
            self.fail(f"HTML processing test failed: {e}")
    
    def test_csv_processing(self):
        """Test CSV processing pipeline"""
        try:
            # Mock for testing purposes
            def process_csv(file_path, text_columns=None, id_column=None, chunk_size=5):
                return [{"text": "col1: value1 | col2: value2", "metadata": {"source": file_path, "rows": [1, 2]}}]
            
            # Create test CSV or use sample
            test_csv_path = os.path.join(CONFIG["paths"]["test_data_dir"], "test_data.csv")
            if not os.path.exists(test_csv_path):
                logger.warning(f"Test CSV not found at {test_csv_path}. Test will use mock data.")
            
            # Process test CSV
            chunks = process_csv(test_csv_path)
            
            # Validate results
            self.assertIsInstance(chunks, list)
            if chunks:
                self.assertDocumentFormat(chunks[0])
                
        except Exception as e:
            logger.error(f"Error in CSV processing test: {e}")
            self.fail(f"CSV processing test failed: {e}")
    
    def test_url_processing(self):
        """Test URL content processing pipeline"""
        try:
            # Mock for testing purposes
            def process_web_url(url, use_selenium=False, delay=1):
                return [{"text": "Web content", "metadata": {"source": url, "document_type": "web"}}]
            
            # Test URL
            test_url = "https://example.com"
            
            # Process test URL
            chunks = process_web_url(test_url)
            
            # Validate results
            self.assertIsInstance(chunks, list)
            if chunks:
                self.assertDocumentFormat(chunks[0])
                
        except Exception as e:
            logger.error(f"Error in URL processing test: {e}")
            self.fail(f"URL processing test failed: {e}")


class RetrievalSystemTests(RAGTestCase):
    """Tests for the retrieval system components"""
    
    def test_vector_database(self):
        """Test vector database setup and search"""
        try:
            # Mock for testing purposes
            def generate_embeddings(texts):
                return np.random.randn(len(texts), 384).astype(np.float32)
            
            def query_vector_database(query_text, top_k=5):
                return [
                    {"rank": i+1, "distance": 0.9-i*0.1, "metadata": {"source": "test"}}
                    for i in range(min(top_k, 3))
                ]
            
            # Test query
            test_query = "What is retrieval augmented generation?"
            
            # Generate embedding for test texts
            test_texts = ["This is a test document", "Another test document"]
            embeddings = generate_embeddings(test_texts)
            
            # Validate embeddings
            self.assertEqual(embeddings.shape, (2, 384))
            self.assertEqual(embeddings.dtype, np.float32)
            
            # Test search functionality
            results = query_vector_database(test_query)
            
            # Validate results
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0)
            for result in results:
                self.assertIn("rank", result)
                self.assertIn("distance", result)
                self.assertIn("metadata", result)
                
        except Exception as e:
            logger.error(f"Error in vector database test: {e}")
            self.fail(f"Vector database test failed: {e}")
    
    def test_bm25_search(self):
        """Test BM25 keyword search implementation"""
        try:
            # Mock for testing purposes
            def search_bm25(query, top_k=5):
                return [
                    {"rank": i+1, "score": 0.9-i*0.1, "metadata": {"source": "test"}}
                    for i in range(min(top_k, 3))
                ]
            
            # Test query
            test_query = "What is retrieval augmented generation?"
            
            # Test search functionality
            results = search_bm25(test_query)
            
            # Validate results
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0)
            for result in results:
                self.assertIn("rank", result)
                self.assertIn("score", result)
                self.assertIn("metadata", result)
                
        except Exception as e:
            logger.error(f"Error in BM25 search test: {e}")
            self.fail(f"BM25 search test failed: {e}")
    
    def test_hybrid_search(self):
        """Test hybrid search system"""
        try:
            # Mock for testing purposes
            class HybridSearchSystem:
                def hybrid_search(self, query):
                    return [
                        {"score": 0.9-i*0.1, "metadata": {"source": "test"}, 
                         "sources": ["Vector Score: 0.85", "BM25 Score: 0.75"]}
                        for i in range(3)
                    ]
            
            # Initialize mock search system
            search_system = HybridSearchSystem()
            
            # Test query
            test_query = "What is retrieval augmented generation?"
            
            # Test hybrid search
            results = search_system.hybrid_search(test_query)
            
            # Validate results
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0)
            for result in results:
                self.assertIn("score", result)
                self.assertIn("metadata", result)
                self.assertIn("sources", result)
                
        except Exception as e:
            logger.error(f"Error in hybrid search test: {e}")
            self.fail(f"Hybrid search test failed: {e}")


class ResponseGenerationTests(RAGTestCase):
    """Tests for the response generation components"""
    
    def test_llm_integration(self):
        """Test LLM integration"""
        try:
            # Mock for testing purposes
            class LLMIntegrationSystem:
                def generate_response(self, query, context_docs, chat_history=None):
                    return {
                        "response": f"This is a response to: {query}",
                        "metadata": {"citations": {}, "query": query}
                    }
            
            # Initialize mock LLM system
            llm_system = LLMIntegrationSystem()
            
            # Test query and context
            test_query = "What is retrieval augmented generation?"
            test_context = [
                {"text": "RAG is a technique that combines retrieval with generation", 
                 "metadata": {"source": "test"}}
            ]
            
            # Generate response
            response_data = llm_system.generate_response(test_query, test_context)
            
            # Validate response
            self.assertIsInstance(response_data, dict)
            self.assertIn("response", response_data)
            self.assertIn("metadata", response_data)
            self.assertTrue(len(response_data["response"]) > 0)
                
        except Exception as e:
            logger.error(f"Error in LLM integration test: {e}")
            self.fail(f"LLM integration test failed: {e}")
    
    def test_citation_system(self):
        """Test citation system"""
        try:
            # Mock for testing purposes
            class CitationManager:
                def __init__(self, citation_style="inline"):
                    self.citation_style = citation_style
                
                def process_sources(self, sources):
                    return {
                        "citations": [
                            {"id": f"src{i}", "marker": f"[{i+1}]", "text": f"Source {i+1}"}
                            for i in range(len(sources))
                        ],
                        "style": self.citation_style
                    }
                
                def format_response_with_citations(self, response_text, citations):
                    return {
                        "text": response_text + " [1][2]",
                        "citations": citations["citations"]
                    }
                
                def verify_citations(self, response_text, citations):
                    return {"verified": True, "results": []}
            
            # Initialize citation manager
            citation_manager = CitationManager()
            
            # Test sources and response
            test_sources = [
                {"text": "RAG combines retrieval with generation", 
                 "metadata": {"source": "test1", "title": "RAG Paper"}},
                {"text": "RAG improves factuality", 
                 "metadata": {"source": "test2", "title": "AI Techniques"}}
            ]
            test_response = "RAG is a technique that enhances LLMs."
            
            # Process sources
            citations = citation_manager.process_sources(test_sources)
            
            # Format response with citations
            formatted_response = citation_manager.format_response_with_citations(
                test_response, citations)
            
            # Verify citations
            verification = citation_manager.verify_citations(
                formatted_response["text"], citations["citations"])
            
            # Validate results
            self.assertIsInstance(citations, dict)
            self.assertIn("citations", citations)
            self.assertIsInstance(formatted_response, dict)
            self.assertIn("text", formatted_response)
            self.assertIsInstance(verification, dict)
            self.assertIn("verified", verification)
                
        except Exception as e:
            logger.error(f"Error in citation system test: {e}")
            self.fail(f"Citation system test failed: {e}")
    
    def test_query_routing(self):
        """Test query routing logic"""
        try:
            # Mock for testing purposes
            class QueryRouter:
                def classify_query(self, query, conversation_context=None):
                    return {
                        "query_type": "unstructured",
                        "query_intent": "informational",
                        "confidence": 0.85,
                        "is_followup": False
                    }
                
                def route_query(self, query, conversation_context=None, metadata=None):
                    return {
                        "success": True,
                        "query_info": self.classify_query(query),
                        "query_plan": {
                            "handler": "retrieval_search",
                            "parameters": {"search_strategy": "hybrid", "top_k": 5}
                        }
                    }
            
            # Initialize query router
            router = QueryRouter()
            
            # Test queries
            test_queries = [
                "What is retrieval augmented generation?",
                "How does vector search work?",
                "Compare BM25 and semantic search"
            ]
            
            # Route each query
            for query in test_queries:
                routing_result = router.route_query(query)
                
                # Validate result
                self.assertIsInstance(routing_result, dict)
                self.assertIn("success", routing_result)
                self.assertIn("query_info", routing_result)
                self.assertIn("query_plan", routing_result)
                self.assertTrue(routing_result["success"])
                
        except Exception as e:
            logger.error(f"Error in query routing test: {e}")
            self.fail(f"Query routing test failed: {e}")


class EvaluationFrameworkTests(RAGTestCase):
    """Tests for the evaluation framework"""
    
    def test_ragas_metrics(self):
        """Test RAGAS metrics implementation"""
        try:
            # Mock for testing purposes
            class RAGASEvaluator:
                def create_evaluation_dataset(self, questions, generated_answers, 
                                             contexts, ground_truths=None):
                    return {
                        "questions": questions,
                        "answers": generated_answers,
                        "contexts": contexts,
                        "ground_truths": ground_truths or []
                    }
                
                def run_evaluation(self, metrics=None):
                    return {
                        "faithfulness": {"mean": 0.85, "std": 0.1},
                        "answer_relevancy": {"mean": 0.78, "std": 0.15},
                        "context_precision": {"mean": 0.82, "std": 0.12},
                        "context_recall": {"mean": 0.75, "std": 0.18},
                        "metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "num_samples": 5
                        }
                    }
            
            # Initialize evaluator
            evaluator = RAGASEvaluator()
            
            # Create sample data
            questions = ["What is RAG?", "How does vector search work?"]
            answers = ["RAG combines retrieval with generation.", 
                      "Vector search uses embedding similarity."]
            contexts = [
                ["RAG is a technique that enhances LLMs."],
                ["Vector search works by finding similar vectors."]
            ]
            
            # Create evaluation dataset
            dataset = evaluator.create_evaluation_dataset(
                questions, answers, contexts)
            
            # Run evaluation
            results = evaluator.run_evaluation()
            
            # Validate results
            self.assertIsInstance(results, dict)
            self.assertIn("faithfulness", results)
            self.assertIn("answer_relevancy", results)
            self.assertIn("context_precision", results)
            self.assertIn("context_recall", results)
            self.assertIn("metadata", results)
            
            # Check metric format
            for metric in ["faithfulness", "answer_relevancy", 
                          "context_precision", "context_recall"]:
                self.assertIn("mean", results[metric])
                self.assertIn("std", results[metric])
                self.assertGreaterEqual(results[metric]["mean"], 0)
                self.assertLessEqual(results[metric]["mean"], 1)
                
        except Exception as e:
            logger.error(f"Error in RAGAS metrics test: {e}")
            self.fail(f"RAGAS metrics test failed: {e}")


class IntegrationTests(RAGTestCase):
    """End-to-end integration tests for the RAG system"""
    
    def test_basic_rag_pipeline(self):
        """Test the basic RAG pipeline from query to response"""
        try:
            # This test would integrate the actual components
            # For demonstration, we'll use mock implementations
            
            # 1. Mock query routing
            def route_query(query):
                return {
                    "query_type": "unstructured",
                    "query_plan": {"search_strategy": "hybrid", "top_k": 3}
                }
            
            # 2. Mock hybrid search
            def hybrid_search(query, top_k=3):
                return [
                    {"score": 0.9, "metadata": {"source": "test1"}, 
                     "text": "RAG combines retrieval with generation."},
                    {"score": 0.8, "metadata": {"source": "test2"}, 
                     "text": "RAG improves factuality of responses."}
                ]
            
            # 3. Mock LLM response generation
            def generate_response(query, context_docs):
                combined_context = " ".join([doc["text"] for doc in context_docs])
                return {
                    "response": f"Based on the context, RAG is a technique that combines retrieval with generation and improves factuality.",
                    "metadata": {"query": query, "context_count": len(context_docs)}
                }
            
            # 4. Mock citation generation
            def add_citations(response_data, context_docs):
                return {
                    "text": response_data["response"] + " [1][2]",
                    "citations": [
                        {"id": "src1", "marker": "[1]", "text": "Source 1"},
                        {"id": "src2", "marker": "[2]", "text": "Source 2"}
                    ]
                }
            
            # Test query
            test_query = "What is retrieval augmented generation?"
            
            # Execute pipeline
            routing_result = route_query(test_query)
            search_results = hybrid_search(test_query, 
                                          routing_result["query_plan"]["top_k"])
            response = generate_response(test_query, search_results)
            final_response = add_citations(response, search_results)
            
            # Validate integration
            self.assertIsInstance(routing_result, dict)
            self.assertIsInstance(search_results, list)
            self.assertIsInstance(response, dict)
            self.assertIsInstance(final_response, dict)
            self.assertIn("text", final_response)
            self.assertIn("citations", final_response)
            self.assertGreater(len(final_response["text"]), 0)
            self.assertGreater(len(final_response["citations"]), 0)
            
        except Exception as e:
            logger.error(f"Error in basic RAG pipeline test: {e}")
            self.fail(f"Basic RAG pipeline test failed: {e}")
    
    def test_conversation_context(self):
        """Test RAG with conversation context handling"""
        try:
            # Mock conversation context
            conversation_context = [
                {"role": "user", "content": "What is RAG?", 
                 "query_type": "unstructured"},
                {"role": "assistant", "content": "RAG combines retrieval with generation."}
            ]
            
            # Mock follow-up detection
            def is_followup_query(query, context):
                return "tell me more" in query.lower() or len(query.split()) < 4
            
            # Mock context-aware response generation
            def generate_context_aware_response(query, context, search_results):
                return {
                    "response": f"Following up on RAG, it also improves factuality of responses.",
                    "metadata": {"is_followup": True, "original_query": context[0]["content"]}
                }
            
            # Test follow-up query
            test_query = "Tell me more about it"
            
            # Execute conversation-aware pipeline
            is_followup = is_followup_query(test_query, conversation_context)
            response = generate_context_aware_response(test_query, 
                                                     conversation_context, [])
            
            # Validate conversation handling
            self.assertTrue(is_followup)
            self.assertIsInstance(response, dict)
            self.assertIn("response", response)
            self.assertIn("metadata", response)
            self.assertTrue(response["metadata"]["is_followup"])
            
        except Exception as e:
            logger.error(f"Error in conversation context test: {e}")
            self.fail(f"Conversation context test failed: {e}")


class PerformanceTests(RAGTestCase):
    """Performance tests for the RAG system"""
    
    def test_ingestion_performance(self):
        """Test ingestion pipeline performance"""
        if not CONFIG["test_settings"]["enable_performance_tests"]:
            self.skipTest("Performance tests disabled")
            
        try:
            # Mock ingestion function with timing
            def process_document(doc_path):
                time.sleep(0.1)  # Simulate processing time
                return [{"text": "Test content", "metadata": {"source": doc_path}}]
            
            # Test with multiple documents
            test_paths = [f"test_doc_{i}.txt" for i in range(10)]
            
            # Measure processing time
            start_time = time.time()
            
            # Process documents
            all_chunks = []
            for path in tqdm(test_paths, desc="Processing documents"):
                chunks = process_document(path)
                all_chunks.extend(chunks)
            
            # Calculate throughput
            total_time = time.time() - start_time
            docs_per_second = len(test_paths) / total_time
            
            logger.info(f"Ingestion performance: {docs_per_second:.2f} docs/second")
            
            # No specific assertions, just logging performance metrics
            
        except Exception as e:
            logger.error(f"Error in ingestion performance test: {e}")
            self.fail(f"Ingestion performance test failed: {e}")
    
    def test_retrieval_performance(self):
        """Test retrieval system performance"""
        if not CONFIG["test_settings"]["enable_performance_tests"]:
            self.skipTest("Performance tests disabled")
            
        try:
            # Mock retrieval function with timing
            def search_documents(query, top_k=5):
                time.sleep(0.2)  # Simulate search time
                return [{"score": 0.9-i*0.1, "metadata": {"source": f"doc_{i}"}} 
                        for i in range(top_k)]
            
            # Test queries
            test_queries = [f"Test query {i}" for i in range(5)]
            
            # Measure search time
            query_times = []
            
            for query in tqdm(test_queries, desc="Testing retrieval"):
                start_time = time.time()
                results = search_documents(query)
                query_time = time.time() - start_time
                query_times.append(query_time)
            
            # Calculate average search time
            avg_search_time = sum(query_times) / len(query_times)
            
            logger.info(f"Retrieval performance: {avg_search_time:.3f} seconds/query")
            
            # No specific assertions, just logging performance metrics
            
        except Exception as e:
            logger.error(f"Error in retrieval performance test: {e}")
            self.fail(f"Retrieval performance test failed: {e}")
    
    def test_end_to_end_performance(self):
        """Test end-to-end RAG system performance"""
        if not CONFIG["test_settings"]["enable_performance_tests"]:
            self.skipTest("Performance tests disabled")
            
        try:
            # Mock end-to-end function with timing
            def process_query(query):
                time.sleep(0.5)  # Simulate total processing time
                return {
                    "response": f"Response to: {query}",
                    "metadata": {"processing_time": 0.5}
                }
            
            # Test queries
            test_queries = [
                "What is retrieval augmented generation?",
                "How does vector search work?",
                "Explain the BM25 algorithm"
            ]
            
            # Measure end-to-end time
            query_times = []
            
            for query in tqdm(test_queries, desc="End-to-end testing"):
                start_time = time.time()
                response = process_query(query)
                query_time = time.time() - start_time
                query_times.append(query_time)
            
            # Calculate average end-to-end time
            avg_time = sum(query_times) / len(query_times)
            
            logger.info(f"End-to-end performance: {avg_time:.3f} seconds/query")
            
            # No specific assertions, just logging performance metrics
            
        except Exception as e:
            logger.error(f"Error in end-to-end performance test: {e}")
            self.fail(f"End-to-end performance test failed: {e}")


def generate_test_report(test_results: Dict[str, Any], output_path: str = None):
    """Generate a comprehensive test report"""
    report = {
        "summary": {
            "total_tests": test_results.get("total", 0),
            "passed": test_results.get("passed", 0),
            "failed": test_results.get("failed", 0),
            "skipped": test_results.get("skipped", 0),
            "pass_rate": test_results.get("pass_rate", 0),
            "execution_time": test_results.get("execution_time", 0)
        },
        "tests": test_results.get("tests", []),
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "system_config": {
                # Add system configuration details here
            }
        },
        "performance_metrics": test_results.get("performance", {})
    }
    
    # Save report as JSON
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
    
    # Print summary to console
    print("\n" + "=" * 50)
    print("TEST EXECUTION SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Skipped: {report['summary']['skipped']}")
    print(f"Pass Rate: {report['summary']['pass_rate']:.2f}%")
    print(f"Total Execution Time: {report['summary']['execution_time']:.2f} seconds")
    print("=" * 50)
    
    return report


def run_tests(test_suites=None):
    """Run specified test suites or all tests"""
    if test_suites is None:
        # Include all test suites
        test_suites = [
            DataIngestionTests,
            RetrievalSystemTests,
            ResponseGenerationTests,
            EvaluationFrameworkTests
        ]
        
        # Add integration tests if enabled
        if CONFIG["test_settings"]["enable_integration_tests"]:
            test_suites.append(IntegrationTests)
            
        # Add performance tests if enabled
        if CONFIG["test_settings"]["enable_performance_tests"]:
            test_suites.append(PerformanceTests)
    
    # Set up test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    for test_class in test_suites:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Set up test runner
    runner = unittest.TextTestRunner(verbosity=2 if CONFIG["reporting"]["verbose_output"] else 1)
    
    # Track overall execution time
    start_time = time.time()
    
    # Run the tests
    print("\n" + "=" * 50)
    print("RUNNING RAG SYSTEM TESTS")
    print("=" * 50)
    
    test_result = runner.run(suite)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Collect test results
    test_results = {
        "total": test_result.testsRun,
        "passed": test_result.testsRun - len(test_result.errors) - len(test_result.failures) - len(test_result.skipped),
        "failed": len(test_result.errors) + len(test_result.failures),
        "skipped": len(test_result.skipped),
        "pass_rate": (test_result.testsRun - len(test_result.errors) - len(test_result.failures) - len(test_result.skipped)) / test_result.testsRun * 100 if test_result.testsRun > 0 else 0,
        "execution_time": execution_time,
        "tests": []
    }
    
    # Add individual test results
    for test_case, error in test_result.errors + test_result.failures:
        test_results["tests"].append({
            "name": str(test_case),
            "status": "failed",
            "error": error,
            "module": test_case.__class__.__module__
        })
    
    # Generate and save test report
    if CONFIG["reporting"]["save_test_results"]:
        report_path = os.path.join(CONFIG["paths"]["test_output_dir"], f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report = generate_test_report(test_results, report_path)
        logger.info(f"Test report saved to {report_path}")
    else:
        report = generate_test_report(test_results)
    
    return test_results, report


def generate_html_report(test_results, output_path=None):
    """Generate an HTML report from test results"""
    if not CONFIG["reporting"]["generate_html_report"]:
        return None
        
    if output_path is None:
        output_path = os.path.join(CONFIG["paths"]["test_output_dir"], f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    
    # Create a simple HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG System Test Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333366; }}
            .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .passed {{ color: green; }}
            .failed {{ color: red; }}
            .skipped {{ color: orange; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>RAG System Test Report</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>Total Tests: {test_results['total']}</p>
            <p>Passed: <span class="passed">{test_results['passed']}</span></p>
            <p>Failed: <span class="failed">{test_results['failed']}</span></p>
            <p>Skipped: <span class="skipped">{test_results['skipped']}</span></p>
            <p>Pass Rate: {test_results['pass_rate']:.2f}%</p>
            <p>Total Execution Time: {test_results['execution_time']:.2f} seconds</p>
        </div>
        
        <h2>Test Details</h2>
        <table>
            <tr>
                <th>Test Name</th>
                <th>Status</th>
                <th>Module</th>
            </tr>
    """
    
    # Add rows for individual tests
    for test in test_results.get('tests', []):
        status_class = "passed" if test.get('status') == "passed" else "failed" if test.get('status') == "failed" else "skipped"
        html_content += f"""
        <tr>
            <td>{test.get('name', 'Unknown')}</td>
            <td class="{status_class}">{test.get('status', 'Unknown')}</td>
            <td>{test.get('module', 'Unknown')}</td>
        </tr>
        """
    
    # Close HTML
    html_content += """
        </table>
        <p><em>Report generated at: {}</em></p>
    </body>
    </html>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Write HTML to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved to {output_path}")
    return output_path


class TestSuiteBuilder:
    """Helper class to build and run custom test suites"""
    
    def __init__(self):
        """Initialize the test suite builder"""
        self.test_classes = []
    
    def add_test_case(self, test_class):
        """Add a test case class to the suite"""
        self.test_classes.append(test_class)
        return self
    
    def add_data_ingestion_tests(self):
        """Add data ingestion tests"""
        self.test_classes.append(DataIngestionTests)
        return self
    
    def add_retrieval_tests(self):
        """Add retrieval system tests"""
        self.test_classes.append(RetrievalSystemTests)
        return self
    
    def add_response_generation_tests(self):
        """Add response generation tests"""
        self.test_classes.append(ResponseGenerationTests)
        return self
    
    def add_evaluation_tests(self):
        """Add evaluation framework tests"""
        self.test_classes.append(EvaluationFrameworkTests)
        return self
    
    def add_integration_tests(self):
        """Add integration tests"""
        self.test_classes.append(IntegrationTests)
        return self
    
    def add_performance_tests(self):
        """Add performance tests"""
        self.test_classes.append(PerformanceTests)
        return self
    
    def run(self):
        """Run the built test suite"""
        return run_tests(self.test_classes)


def create_sample_test_data():
    """Create sample test data files for testing"""
    # Create test directories
    os.makedirs(CONFIG["paths"]["test_data_dir"], exist_ok=True)
    
    # Create a sample PDF-like file (not a real PDF)
    pdf_path = os.path.join(CONFIG["paths"]["test_data_dir"], "sample_document.pdf")
    with open(pdf_path, 'w', encoding='utf-8') as f:
        f.write("This is sample PDF content for testing purposes.")
    
    # Create a sample HTML file
    html_path = os.path.join(CONFIG["paths"]["test_data_dir"], "sample_webpage.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("<html><head><title>Sample Page</title></head><body><h1>Sample Content</h1><p>This is sample HTML content for testing purposes.</p></body></html>")
    
    # Create a sample CSV file
    csv_path = os.path.join(CONFIG["paths"]["test_data_dir"], "sample_data.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("id,name,value\n1,Item 1,100\n2,Item 2,200\n3,Item 3,300")
    
    # Create a sample URL list
    url_path = os.path.join(CONFIG["paths"]["test_data_dir"], "sample_urls.txt")
    with open(url_path, 'w', encoding='utf-8') as f:
        f.write("https://example.com\nhttps://example.org")
    
    # Create a sample ground truth file
    ground_truth_path = os.path.join(CONFIG["paths"]["test_data_dir"], "sample_ground_truth.json")
    ground_truth = {
        "questions": [
            "What is retrieval augmented generation?",
            "How does vector search work?"
        ],
        "answers": [
            "Retrieval Augmented Generation (RAG) is an AI technique that enhances language models by incorporating external knowledge retrieval.",
            "Vector search works by converting text into high-dimensional vectors using embedding models, then finding similar vectors using distance metrics."
        ]
    }
    with open(ground_truth_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2)
    
    logger.info(f"Created sample test data in {CONFIG['paths']['test_data_dir']}")
    
    return {
        "pdf_path": pdf_path,
        "html_path": html_path,
        "csv_path": csv_path,
        "url_path": url_path,
        "ground_truth_path": ground_truth_path
    }


def main():
    """Main function to run the testing framework"""
    print("\n" + "=" * 50)
    print("RAG SYSTEM TESTING FRAMEWORK")
    print("=" * 50)
    
    # Create sample test data
    print("\nCreating sample test data...")
    sample_data = create_sample_test_data()
    
    # Run all tests
    print("\nRunning tests...")
    test_results, report = run_tests()
    
    # Generate HTML report if enabled
    if CONFIG["reporting"]["generate_html_report"]:
        html_path = generate_html_report(test_results)
        print(f"\nHTML report generated: {html_path}")
    
    # Print final summary
    print("\n" + "=" * 50)
    print("TESTING COMPLETE")
    print("=" * 50)
    print(f"Pass rate: {test_results['pass_rate']:.2f}%")
    print(f"Total execution time: {test_results['execution_time']:.2f} seconds")
    
    return test_results, report


# Run the main function when the script is executed directly
if __name__ == "__main__":
    main()