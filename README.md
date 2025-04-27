# Enhanced RAG System

A comprehensive Retrieval-Augmented Generation system with advanced features and performance optimizations.

## Overview

This project extends a basic RAG pipeline with:

1. **Advanced Features** - Query rewriting, personalization, source reliability scoring, and conversation context
2. **Performance Optimization** - Profiling, memory optimization, caching, and batch processing

## Directory Structure

```
MLT_SRC_QA_System/
├── config/
│   └── config.yaml
├── data/
│   ├── advanced/       # Storage for advanced features
│   ├── processed/      # Processed document chunks
│   ├── vector_db/      # Vector database storage
│   └── bm25_index/     # BM25 index storage
├── docs/
│   └── system_documentation.md
├── modules/            # Core RAG modules
│   ├── bm25_search.py
│   ├── citation_system.py
│   ├── hybrid_search.py
│   ├── llm_integration.py
│   ├── pdf_processing.py
│   ├── query_routing.py
│   └── vector_database.py
├── logs/               # Log files
├── advanced_features.py   # New advanced features module
├── optimization.py        # New optimization module
├── pipeline_integration.py   # Original pipeline
├── system_integration.py     # Enhanced pipeline integration
└── test_enhanced_rag.py      # Test script
```

## Components

### Advanced Features

#### Query Rewriting
Improves retrieval by enhancing queries with:
- Expanded abbreviations and synonyms
- More specific terminology
- Context from conversation history

#### Source Reliability Scoring
Evaluates source trustworthiness based on:
- Source type (PDF, academic, webpage)
- Author information
- Domain reputation
- Publication date

#### Personalization
Customizes responses using:
- User topic interests
- Technical level preferences
- Document interaction history

#### Conversation Context
Maintains conversational continuity:
- Tracks conversation history
- Resolves references to previous topics
- Provides context for ambiguous queries

### Optimization

#### Performance Profiling
Measures and analyzes system performance:
- Component-level timing
- Memory usage tracking
- Bottleneck identification

#### Memory Optimization
Reduces memory footprint for large document collections:
- Optimized document storage
- Efficient embedding representation

#### Caching
Avoids redundant processing:
- Query result caching
- Embedding caching
- Time-based invalidation

#### Batch Processing
Processes multiple operations efficiently:
- Batched document embedding
- Parallel query processing
- Optimized batch sizes

## Usage

### Basic Usage

```python
from system_integration import EnhancedRAGPipeline

# Initialize pipeline
pipeline = EnhancedRAGPipeline()

# Process a query
response = pipeline.enhanced_process_query(
    "What is retrieval augmented generation?",
    user_id="example_user"
)

# Access the response
print(response['response'])

# Access metadata
print(response['metadata'])
```

### Batch Processing

```python
from system_integration import EnhancedRAGPipeline

# Initialize pipeline
pipeline = EnhancedRAGPipeline()

# Process multiple queries
queries = [
    "What is vector search?",
    "How does BM25.work?",
    "Explain hybrid search"
]

# Process in batch
results = pipeline.batch_process_queries(queries, user_id="example_user")
```

### Personalization Example

```python
from system_integration import EnhancedRAGPipeline

# Initialize pipeline
pipeline = EnhancedRAGPipeline()

# Process queries for the same user to see personalization effects
user_id = "john_doe"

# First query establishes interests
pipeline.enhanced_process_query("Tell me about vector databases", user_id=user_id)

# Second query will be personalized based on first query
result = pipeline.enhanced_process_query("What embedding models are best?", user_id=user_id)
```

## Testing

Run the comprehensive test script:

```
python test_enhanced_rag.py
```

This tests all aspects of the enhanced system:
- Basic query processing
- Personalization
- Batch processing
- Conversation context
- Source reliability scoring
- Optimization statistics

## Configuration

Configuration is handled through `config/config.yaml`. Key settings:

```yaml
optimization:
  max_cache_size_mb: 1000
  batch_size: 16
  max_workers: 4
  enable_profiling: true
  enable_caching: true
  enable_memory_optimization: true
  enable_batch_processing: true

advanced_features:
  enable_query_rewriting: true
  enable_source_reliability: true
  enable_personalization: true
  enable_conversation_context: true
```

## Future Enhancements

Potential next steps:
- Web UI for testing and demonstration
- Extended evaluation framework
- Feedback collection mechanism
- Additional embedding models
