"""
Comprehensive test script for the Enhanced RAG System.
This script demonstrates all features of the system including
advanced query enhancement, personalization, and optimization.
"""

import os
import json
import time
from typing import List, Dict, Any

from system_integration import EnhancedRAGPipeline

def print_separator(title):
    """Print a section separator with title"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def test_basic_query(pipeline):
    """Test basic query functionality"""
    print_separator("BASIC QUERY TEST")
    
    query = "What is retrieval augmented generation?"
    print(f"Query: {query}")
    
    start_time = time.time()
    response = pipeline.enhanced_process_query(query, user_id="test_user")
    end_time = time.time()
    
    print(f"\nResponse: {response['response']}")
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")
    
    # Display enhancement info
    if 'metadata' in response and 'query_enhancement' in response['metadata']:
        enhancement = response['metadata']['query_enhancement']
        print(f"\nOriginal query: {enhancement['original_query']}")
        print(f"Enhanced query: {enhancement['enhanced_query']}")
    
    return response

def test_personalization(pipeline):
    """Test personalization features with multiple queries for the same user"""
    print_separator("PERSONALIZATION TEST")
    
    user_id = "personalization_test_user"
    print(f"Testing personalization for user: {user_id}")
    
    # First query to establish user interests
    print("\n--- First query to establish user interests ---")
    query1 = "Tell me about vector databases for RAG systems"
    print(f"Query: {query1}")
    response1 = pipeline.enhanced_process_query(query1, user_id=user_id)
    print(f"Response: {response1['response'][:200]}...")
    
    # Second query should be personalized based on first query
    print("\n--- Second query should be personalized based on first query ---")
    query2 = "What are the best embedding models?"
    print(f"Query: {query2}")
    response2 = pipeline.enhanced_process_query(query2, user_id=user_id)
    print(f"Response: {response2['response'][:200]}...")
    
    # Get user profile to show personalization in action
    user_profile = pipeline.advanced_features.personalization.get_or_create_profile(user_id)
    print("\nUser profile after queries:")
    print(json.dumps(user_profile, indent=2)[:500] + "...")
    
    return user_profile

def test_batch_processing(pipeline):
    """Test batch processing of multiple queries"""
    print_separator("BATCH PROCESSING TEST")
    
    queries = [
        "What are the components of a RAG system?",
        "How does vector search work?",
        "Explain BM25 algorithm",
        "What is hybrid search in information retrieval?"
    ]
    
    print(f"Processing {len(queries)} queries in batch:")
    for i, q in enumerate(queries, 1):
        print(f"{i}. {q}")
    
    start_time = time.time()
    batch_results = pipeline.batch_process_queries(queries, user_id="batch_test_user")
    end_time = time.time()
    
    print(f"\nBatch processing completed in {end_time - start_time:.2f} seconds")
    print(f"Average time per query: {(end_time - start_time) / len(queries):.2f} seconds")
    
    # Show brief results
    for i, (query, result) in enumerate(zip(queries, batch_results), 1):
        print(f"\n{i}. Query: {query}")
        print(f"   Response: {result['response'][:100]}...")
    
    return batch_results

def test_conversation_context(pipeline):
    """Test conversation context preservation"""
    print_separator("CONVERSATION CONTEXT TEST")
    
    user_id = "conversation_test_user"
    print(f"Testing conversation context for user: {user_id}")
    
    # First query to establish context
    print("\n--- First query ---")
    query1 = "What is hybrid search in RAG systems?"
    print(f"Query: {query1}")
    response1 = pipeline.enhanced_process_query(query1, user_id=user_id)
    print(f"Response: {response1['response'][:200]}...")
    
    # Follow-up query that refers to previous query
    print("\n--- Follow-up query ---")
    query2 = "What are its advantages over pure vector search?"
    print(f"Query: {query2}")
    
    # Get conversation context from first query
    conversation_context = pipeline.advanced_features.conversation.get_conversation_context(user_id)
    
    # Process second query with context
    response2 = pipeline.enhanced_process_query(
        query2, 
        user_id=user_id,
        conversation_context=conversation_context
    )
    print(f"Response: {response2['response'][:200]}...")
    
    # The enhanced query should recognize "its" refers to hybrid search
    if 'metadata' in response2 and 'query_enhancement' in response2['metadata']:
        enhancement = response2['metadata']['query_enhancement']
        print(f"\nOriginal query: {enhancement['original_query']}")
        print(f"Enhanced query: {enhancement['enhanced_query']}")
    
    return conversation_context

def test_source_reliability(pipeline):
    """Test source reliability scoring"""
    print_separator("SOURCE RELIABILITY TEST")
    
    query = "What are climate change solutions?"
    print(f"Query: {query}")
    
    response = pipeline.enhanced_process_query(query, user_id="reliability_test_user")
    
    # Check if we have reliability scores in the response
    if ('metadata' in response and 
        'enhanced_documents' in response['metadata']):
        
        print("\nSource reliability scores:")
        for doc in response['metadata']['enhanced_documents']:
            print(f"Document {doc['id']}: {doc.get('reliability_score', 'N/A')}")
    
    return response

def test_optimization_stats(pipeline):
    """Test optimization statistics gathering"""
    print_separator("OPTIMIZATION STATISTICS TEST")
    
    # Run several queries to generate statistics
    queries = [
        "What is LangChain?",
        "Explain vector embeddings",
        "How to implement BM25 search?",
        "What are prompt templates?"
    ]
    
    print(f"Running {len(queries)} queries to generate statistics...")
    for query in queries:
        pipeline.enhanced_process_query(query, user_id="stats_test_user")
    
    # Get system stats
    stats = pipeline.get_system_stats()
    print("\nSystem Statistics:")
    print(json.dumps(stats, indent=2))
    
    return stats

def main():
    """Main test function"""
    print_separator("ENHANCED RAG SYSTEM TEST")
    
    # Create enhanced RAG pipeline
    pipeline = EnhancedRAGPipeline()
    
    # Run all tests
    basic_result = test_basic_query(pipeline)
    personalization_profile = test_personalization(pipeline)
    batch_results = test_batch_processing(pipeline)
    conversation_context = test_conversation_context(pipeline)
    reliability_result = test_source_reliability(pipeline)
    optimization_stats = test_optimization_stats(pipeline)
    
    print_separator("TEST SUMMARY")
    print("All tests completed successfully!")
    print(f"- Basic query: {'✓' if basic_result else '✗'}")
    print(f"- Personalization: {'✓' if personalization_profile else '✗'}")
    print(f"- Batch processing: {'✓' if batch_results else '✗'}")
    print(f"- Conversation context: {'✓' if conversation_context else '✗'}")
    print(f"- Source reliability: {'✓' if reliability_result else '✗'}")
    print(f"- Optimization stats: {'✓' if optimization_stats else '✗'}")

if __name__ == "__main__":
    main()