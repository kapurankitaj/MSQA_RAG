import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============= CONFIGURATION SETTINGS =============
CONFIG = {
    # Query type classification
    "classification": {
        "default_query_type": "unstructured",
        "similarity_threshold": 0.6,
        "use_sql_keywords": True,
        "use_keyword_boosting": True,
        "hybrid_classification_boost": 0.2
    },
    
    # SQL query generation
    "sql": {
        "generation_temperature": 0.3,
        "max_tokens": 200,
        "safety_checks": True
    },
    
    # Hybrid retrieval
    "retrieval": {
        "vector_weight": 0.7,
        "keyword_weight": 0.3,
        "top_k": 5,
        "diversity_factor": 0.2
    },
    
    # Fallback strategy
    "fallback": {
        "max_retry_attempts": 2,
        "retrieval_increase": 3,
        "enable_query_enhancement": True,
        "min_confidence_threshold": 0.4
    },
    
    # Logging
    "logging": {
        "level": logging.INFO,
        "format": '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        "log_file": "query_routing.log"
    }
}

# Set up logging
logging.basicConfig(
    level=CONFIG["logging"]["level"],
    format=CONFIG["logging"]["format"],
    handlers=[
        logging.FileHandler(CONFIG["logging"]["log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QueryRouting")

class QueryType(Enum):
    """Enum representing different types of queries that can be handled."""
    STRUCTURED = "structured"  # SQL or structured data queries
    UNSTRUCTURED = "unstructured"  # General text-based queries
    HYBRID = "hybrid"  # Queries requiring both structured and unstructured data
    CONVERSATIONAL = "conversational"  # Follow-up or context-dependent queries
    CALCULATION = "calculation"  # Mathematical or numerical calculation queries
    UNKNOWN = "unknown"  # Unknown query type (will use fallback)

class QueryIntent(Enum):
    """Enum representing the intent behind a query."""
    INFORMATIONAL = "informational"  # Seeking facts or information
    ANALYTICAL = "analytical"  # Seeking analysis or insights
    COMPARATIVE = "comparative"  # Comparing entities or concepts
    PROCEDURAL = "procedural"  # Seeking instructions or steps
    EXPLORATORY = "exploratory"  # Open-ended exploration
    CLARIFICATION = "clarification"  # Seeking clarification
    UNKNOWN = "unknown"  # Unknown intent

class QueryRouter:
    """
    Routes queries to the appropriate handler based on query classification.
    Implements decision tree logic for determining how to process different queries.
    """
    
    def __init__(
        self,
        vector_search_fn: Optional[Callable] = None,
        keyword_search_fn: Optional[Callable] = None, 
        sql_query_fn: Optional[Callable] = None,
        llm_query_fn: Optional[Callable] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Query Router
        
        Args:
            vector_search_fn: Function to perform vector similarity search
            keyword_search_fn: Function to perform keyword search
            sql_query_fn: Function to execute SQL queries
            llm_query_fn: Function to query LLM directly
            config: Optional configuration dictionary to override defaults
        """
        self.vector_search_fn = vector_search_fn
        self.keyword_search_fn = keyword_search_fn
        self.sql_query_fn = sql_query_fn
        self.llm_query_fn = llm_query_fn
        
        # Load configuration
        self.config = CONFIG.copy()
        
        # Update with custom config if provided
        if config:
            # Deep merge configuration
            self._merge_config(self.config, config)
        
        # Initialize classifier components
        self._init_classifiers()
        
        logger.info("Query Router initialized with configuration")
    
    def _merge_config(self, base_config, new_config):
        """Deep merge configurations"""
        for key, value in new_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def _init_classifiers(self):
        """Initialize all classification components."""
        # Initialize text vectorizer for similarity-based classification
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        
        # Expanded and more diverse example queries for each type
        self.example_queries = {
            QueryType.STRUCTURED: [
                "What is the total revenue for Q1 2023?",
                "Show me sales by region for last month",
                "Count the number of customers in California",
                "List all products with inventory below 100 units",
                "What is the average transaction value by customer segment?",
                "Show me the top 5 performing stores",
                "How many orders were placed yesterday?",
                "Calculate the growth rate between 2022 and 2023",
                "What percentage of customers are in the premium tier?",
                "Give me a breakdown of expenses by department"
            ],
            QueryType.UNSTRUCTURED: [
                "Explain how retrieval-augmented generation works",
                "What are the key benefits of vector databases?",
                "Summarize the main points of the latest research paper",
                "What are best practices for prompt engineering?",
                "Describe the process of fine-tuning language models",
                "What challenges are associated with implementing RAG systems?",
                "Who developed the original transformer architecture?",
                "What is the difference between semantic search and keyword search?",
                "Explain the concept of embedding vectors",
                "How does BM25 algorithm work?"
            ],
            QueryType.HYBRID: [
                # Business Performance Integration
                "Compare Q3 sales performance with market growth projections",
                "Analyze customer retention rates in context of marketing spend",
                "Relate product development costs to revenue generation",
                "Evaluate team productivity metrics against industry benchmarks",
                
                # Multi-dimensional Analysis
                "How do customer demographics correlate with purchasing patterns?",
                "Connect support ticket volumes with product feature adoption",
                "Interpret financial results in light of economic indicators",
                "Bridge customer feedback with product development priorities",
                
                # Cross-domain Insights
                "What insights from user research apply to our product strategy?",
                "Synthesize sales data with customer sentiment analysis",
                "Understand team performance through multiple performance metrics",
                "Evaluate strategic objectives considering market constraints",
                
                # Nuanced Analytical Queries
                "How do internal efficiency metrics reflect broader market trends?",
                "Contextualize our innovation spending with competitive landscape",
                "Integrate qualitative feedback with quantitative performance data",
                "Explore the relationship between R&D investment and market share"
            ],
            QueryType.CONVERSATIONAL: [
                "Tell me more about that",
                "Why is that the case?",
                "Can you elaborate on the previous point?",
                "What else should I know about this?",
                "How does that compare to the earlier example?",
                "Could you explain that differently?",
                "What are the implications of this?",
                "Is there another way to look at this?",
                "How certain are you about this information?",
                "Can you provide more examples?"
            ],
            QueryType.CALCULATION: [
                "Calculate the ROI for a $10,000 investment with 7% annual return over 5 years",
                "What is the compound annual growth rate if we went from $1M to $1.5M in 3 years?",
                "If customer acquisition cost is $50 and lifetime value is $300, what's the LTV:CAC ratio?",
                "Calculate the standard deviation of the following values: 10, 12, 23, 23, 16, 23, 21, 16",
                "What's the present value of $5000 received in 10 years with a 5% discount rate?",
                "If we have 25% market share of a $80M market, what's our estimated revenue?",
                "Calculate the break-even point if fixed costs are $50,000 and contribution margin is 35%",
                "What sample size do we need for a 95% confidence level with a 5% margin of error?",
                "If our conversion rate increased from 2.3% to 3.1%, what's the percentage improvement?",
                "Calculate the expected value if we have a 30% chance of $100K profit and 70% chance of $20K loss"
            ]
        }
        
        # SQL-specific keywords for detection
        self.sql_keywords = [
            "select", "from", "where", "join", "group by", "order by", 
            "having", "table", "database", "query", "rows", "columns",
            "count", "sum", "average", "avg", "maximum", "max", "minimum", "min",
            "total", "percentage", "ratio", "compare", "filter", "sort"
        ]
        
        # Hybrid keywords with stronger semantic indicators
        self.hybrid_boost_keywords = [
            # Comparative and integrative language
            "compare", "relate", "connect", "integrate", "correlate", "contextualize",
            "alongside", "in context of", "with respect to", "in relation to", 
            
            # Cross-domain analysis indicators
            "across", "between", "intersection", "relationship", "link", 
            "tie", "connect", "bridge", "synthesize", "combine",
            
            # Analytical integration phrases
            "how does", "what is the connection", "analyze in context",
            "interpret in light of", "understand through", "evaluate together",
            
            # Specific domain bridging phrases
            "market insights with", "financial data and", "customer feedback versus",
            "performance metrics in light of", "strategic objectives compared to",
            
            # Nuanced analytical language
            "underlying factors", "broader context", "holistic view", 
            "comprehensive analysis", "multidimensional perspective"
        ]
        
        # Fit vectorizer on example queries
        all_examples = []
        for query_type, examples in self.example_queries.items():
            all_examples.extend(examples)
        
        self.vectorizer.fit(all_examples)
        
        # Create vectorized examples
        self.example_vectors = {}
        for query_type, examples in self.example_queries.items():
            self.example_vectors[query_type] = self.vectorizer.transform(examples)
    
    def _detect_hybrid_query(self, query: str) -> float:
        """
        Specialized method to detect potential hybrid queries
        
        Returns:
            Confidence score for hybrid query classification
        """
        hybrid_indicators = [
            r'compare.*with',
            r'relate.*to',
            r'in\s+context\s+of',
            r'alongside',
            r'connect.*and',
            r'integrate.*with',
            r'reflect.*through',
            r'correlate.*with',
            r'interpret.*in\s+light\s+of',
            r'understand.*through'
        ]
        
        # Preprocess query
        query_lower = query.lower()
        
        # Check for hybrid keywords
        hybrid_keyword_matches = sum(1 for kw in self.hybrid_boost_keywords if kw in query_lower)
        
        # Check for hybrid pattern matches
        pattern_matches = sum(1 for pattern in hybrid_indicators if re.search(pattern, query, re.IGNORECASE))
        
        # Calculate hybrid confidence
        if hybrid_keyword_matches >= 2 and pattern_matches >= 1:
            return 0.85  # Very high confidence
        elif hybrid_keyword_matches >= 1 and pattern_matches >= 1:
            return 0.7  # High confidence
        elif hybrid_keyword_matches >= 2 or pattern_matches >= 2:
            return 0.6  # Moderate confidence
        
        return 0.0  # Not a hybrid query
    
    def classify_query(
        self, 
        query: str,
        conversation_context: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Classify a query to determine its type and intent
        
        Args:
            query: The user's query string
            conversation_context: Optional list of prior conversation turns
            
        Returns:
            Dictionary with classification results
        """
        # Preprocess the query
        processed_query = self._preprocess_query(query)
        
        # Detect hybrid query first
        hybrid_confidence = self._detect_hybrid_query(processed_query)
        if hybrid_confidence > 0.5:
            return {
                "query_type": QueryType.HYBRID,
                "query_intent": self._determine_intent(query),
                "confidence": hybrid_confidence,
                "is_followup": False
            }
        
        # Check if this is a follow-up query
        is_followup = self._is_followup_query(query, conversation_context)
        
        # If it's a clear follow-up and we have conversation context
        if is_followup and conversation_context:
            return {
                "query_type": QueryType.CONVERSATIONAL,
                "query_intent": QueryIntent.CLARIFICATION,
                "confidence": 0.9,
                "is_followup": True,
                "original_query_type": self._get_original_query_type(conversation_context)
            }
        
        # Check for SQL patterns if configured
        if self.config["classification"]["use_sql_keywords"] and self._has_sql_patterns(processed_query):
            return {
                "query_type": QueryType.STRUCTURED,
                "query_intent": self._determine_intent(query),
                "confidence": 0.85,
                "is_followup": False,
                "sql_likelihood": 0.9
            }
        
        # Check for calculation patterns
        if self._is_calculation_query(processed_query):
            return {
                "query_type": QueryType.CALCULATION,
                "query_intent": QueryIntent.ANALYTICAL,
                "confidence": 0.85,
                "is_followup": False
            }
        
        # Vectorize the query
        query_vector = self.vectorizer.transform([processed_query])
        
        # Compare with example vectors to find the closest match
        best_match_type = None
        best_match_score = 0
        
        for query_type, example_vectors in self.example_vectors.items():
            # Calculate similarity with each example of this type
            similarities = cosine_similarity(query_vector, example_vectors).flatten()
            avg_similarity = np.mean(similarities)
            max_similarity = np.max(similarities)
            
            # Modify similarity calculation to be more nuanced
            hybrid_boost = self.config["classification"]["hybrid_classification_boost"] if query_type == QueryType.HYBRID else 0
            weighted_score = (
                0.6 * max_similarity +  # Maximum similarity weight
                0.3 * avg_similarity +  # Average similarity
                0.1 * hybrid_boost      # Slight hybrid bias if configured
            )
            
            if weighted_score > best_match_score:
                best_match_score = weighted_score
                best_match_type = query_type
        
        # Apply keyword boosting if configured
        if self.config["classification"]["use_keyword_boosting"]:
            boosted_score = self._apply_keyword_boosting(processed_query, best_match_type, best_match_score)
            if boosted_score != best_match_score:
                best_match_score = boosted_score
        
        # Check if we have a reliable classification
        if best_match_score >= self.config["classification"]["similarity_threshold"]:
            intent = self._determine_intent(query)
            return {
                "query_type": best_match_type,
                "query_intent": intent,
                "confidence": best_match_score,
                "is_followup": False
            }
        else:
            # Default to unstructured if no clear classification
            return {
                "query_type": QueryType.UNSTRUCTURED,
                "query_intent": self._determine_intent(query),
                "confidence": 0.4,  # Low confidence
                "is_followup": False
            }
    
    def _apply_keyword_boosting(
        self, 
        query: str, 
        current_type: QueryType, 
        current_score: float
    ) -> float:
        """
        Apply keyword boosting to adjust classification scores
        
        Args:
            query: Preprocessed query
            current_type: Current best matching query type
            current_score: Current match score
            
        Returns:
            Adjusted score after boosting
        """
        # Keywords that strongly indicate specific query types
        type_keywords = {
            QueryType.STRUCTURED: [
                "database", "table", "column", "row", "record", "field",
                "query", "select", "from", "where", "count", "sum", 
                "average", "join", "group", "filter", "sort", "order"
            ],
            QueryType.UNSTRUCTURED: [
                "document", "article", "paper", "report", "explain",
                "describe", "summarize", "context", "meaning", "interpret",
                "analyze", "concept", "idea", "theory", "background"
            ],
            QueryType.HYBRID: [
                "compare", "relate", "connect", "integrate", "correlate", 
                "contextualize", "alongside", "in context of", 
                "with respect to", "in relation to"
            ],
            QueryType.CALCULATION: [
                "calculate", "compute", "formula", "equation", "result", 
                "solve", "value", "number", "quantity", "rate", "ratio",
                "percentage", "total", "sum", "difference", "product"
            ]
        }
        
        # Check for hybrid keywords first
        hybrid_matches = sum(1 for kw in type_keywords[QueryType.HYBRID] if kw in query)
        
        # If strong hybrid indicators, boost hybrid classification
        if hybrid_matches > 0:
            hybrid_boost = min(0.3, 0.1 * hybrid_matches)
            if current_type != QueryType.HYBRID:
                return min(0.9, current_score + hybrid_boost)
        
        # Check for keywords matching current type
        if current_type in type_keywords:
            relevant_keywords = type_keywords[current_type]
            matches = sum(1 for kw in relevant_keywords if kw in query)
            
            # Boost score based on keyword matches
            if matches > 0:
                # Calculate boost (diminishing returns for multiple matches)
                boost = min(0.15, 0.05 * matches)
                return min(0.95, current_score + boost)
        
        return current_score
    
    def route_query(
        self, 
        query: str,
        conversation_context: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route a query to the appropriate handler based on classification
        
        Args:
            query: The user's query string
            conversation_context: Optional list of prior conversation turns
            metadata: Optional metadata about the query or user
            
        Returns:
            Dictionary with routing decision and related information
        """
        # Classify the query
        classification_result = self.classify_query(query, conversation_context)
        query_type = classification_result["query_type"]
        confidence = classification_result["confidence"]
        
        logger.info(f"Classified query as {query_type.value} with confidence {confidence:.2f}")
        
        # Check if confidence is too low for reliable classification
        if confidence < self.config["fallback"]["min_confidence_threshold"]:
            logger.warning(f"Classification confidence {confidence:.2f} below threshold {self.config['fallback']['min_confidence_threshold']}")
            # Use fallback routing strategy
            return self._fallback_routing_strategy(query, classification_result, conversation_context, metadata)
        
        # Route based on query type
        if query_type == QueryType.STRUCTURED:
            return self._handle_structured_query(query, classification_result, metadata)
        elif query_type == QueryType.UNSTRUCTURED:
            return self._handle_unstructured_query(query, classification_result, metadata)
        elif query_type == QueryType.HYBRID:
            return self._handle_hybrid_query(query, classification_result, metadata)
        elif query_type == QueryType.CONVERSATIONAL:
            return self._handle_conversational_query(query, classification_result, conversation_context, metadata)
        elif query_type == QueryType.CALCULATION:
            return self._handle_calculation_query(query, classification_result, metadata)
        else:
            # Unknown query type
            return self._fallback_routing_strategy(query, classification_result, conversation_context, metadata)
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess query for classification
        
        Args:
            query: Original query string
            
        Returns:
            Preprocessed query
        """
        # Convert to lowercase
        query = query.lower()
        
        # Remove punctuation
        query = re.sub(r'[^\w\s]', ' ', query)
        
        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query
    
    def _is_followup_query(
        self, 
        query: str,
        conversation_context: Optional[List[Dict[str, str]]]
    ) -> bool:
        """
        Determine if a query is a follow-up to a previous conversation
        
        Args:
            query: Query string
            conversation_context: Previous conversation turns
            
        Returns:
            Boolean indicating if this is likely a follow-up query
        """
        if not conversation_context:
            return False
            
        # Look for explicit follow-up indicators
        followup_phrases = [
            "tell me more", "more detail", "elaborate", "explain further",
            "what about", "how about", "why is that", "can you expand",
            "what does that mean", "could you clarify", "give me examples",
            "and how does", "what if", "continue", "go on", "proceed",
            "furthermore", "additionally", "also", "too", "as well"
        ]
        
        query_lower = query.lower()
        
        # Check for pronoun references (it, they, them, etc.)
        pronoun_pattern = r'\b(it|they|them|their|these|this|that|those)\b'
        has_pronouns = bool(re.search(pronoun_pattern, query_lower))
        
        # Check for explicit follow-up phrases
        has_followup_phrase = any(phrase in query_lower for phrase in followup_phrases)
        
        # Check if query is very short (likely a follow-up)
        is_short_query = len(query.split()) < 4
        
        # Check if lacks context on its own
        lacks_context = is_short_query and not any(kw in query_lower for kw in ["what", "how", "why", "who", "when", "where", "explain", "tell", "show", "give", "find", "search"])
        
        return has_pronouns or has_followup_phrase or lacks_context
    
    def _get_original_query_type(
        self, 
        conversation_context: List[Dict[str, str]]
    ) -> QueryType:
        """
        Get the original query type from conversation context
        
        Args:
            conversation_context: Previous conversation turns
            
        Returns:
            Original query type
        """
        if not conversation_context:
            return QueryType.UNKNOWN
            
        # Look for the most recent non-conversational query
        for turn in reversed(conversation_context):
            if "query_type" in turn and turn["query_type"] != QueryType.CONVERSATIONAL.value:
                try:
                    return QueryType(turn["query_type"])
                except (ValueError, KeyError):
                    pass
                    
        return QueryType.UNKNOWN
    
    def _has_sql_patterns(self, query: str) -> bool:
        """
        Check if a query contains SQL-like patterns
        
        Args:
            query: Preprocessed query string
            
        Returns:
            Boolean indicating if query has SQL patterns
        """
        # Check for SQL keywords
        query_words = set(query.split())
        sql_word_count = sum(1 for word in self.sql_keywords if word in query_words)
        has_multiple_sql_keywords = sql_word_count >= 2
        
        # Check for SQL-like patterns
        sql_patterns = [
            r'select\s+.+\s+from',
            r'show\s+.+\s+from',
            r'table\s+.+\s+where',
            r'group\s+by',
            r'order\s+by',
            r'join',
            r'count\s+.*\s+where',
            r'sum\s+of',
            r'average\s+of',
            r'total\s+.*\s+by'
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, query):
                return True
                
        return has_multiple_sql_keywords
    
    def _is_calculation_query(self, query: str) -> bool:
        """
        Check if a query is asking for a calculation
        
        Args:
            query: Preprocessed query string
            
        Returns:
            Boolean indicating if query is calculation-focused
        """
        # Check for calculation keywords
        calculation_keywords = [
            "calculate", "compute", "sum", "average", "mean", "median", 
            "percentage", "total", "difference", "ratio", "divide", 
            "multiply", "add", "subtract", "increase", "decrease", 
            "growth rate", "compound", "interest", "roi", "return", 
            "profit margin", "standard deviation", "variance", "regression"
        ]
        
        # Check for mathematical symbols
        math_symbols = ["+", "-", "*", "/", "^", "%", "="]
        
        # Check for numbers
        has_numbers = bool(re.search(r'\d+', query))
        
        # Count calculation keywords in query
        keyword_count = sum(1 for kw in calculation_keywords if kw in query)
        
        # Count math symbols in query
        symbol_count = sum(1 for sym in math_symbols if sym in query)
        
        # Stronger signals for calculation queries
        has_calculation_pattern = bool(re.search(r'calculate\s+the', query)) or \
                                  bool(re.search(r'what\s+is\s+the\s+(sum|average|total|difference)', query)) or \
                                  bool(re.search(r'how\s+(much|many)\s+.*\s+if', query))
        
        # Determine if this is likely a calculation query
        return (has_calculation_pattern or 
                (has_numbers and (keyword_count >= 1 or symbol_count >= 1)) or
                (keyword_count >= 2))
    
    def _determine_intent(self, query: str) -> QueryIntent:
        """
        Determine the intent behind a query
        
        Args:
            query: Original query string
            
        Returns:
            QueryIntent enum value
        """
        query_lower = query.lower()
        
        # Check for informational intent (seeking facts)
        info_patterns = [
            r'what\s+is', r'who\s+is', r'when\s+did', r'where\s+is',
            r'tell\s+me\s+about', r'explain', r'describe',
            r'definition\s+of', r'meaning\s+of'
        ]
        
        # Check for analytical intent
        analytical_patterns = [
            r'why\s+is', r'how\s+does', r'what\s+causes', r'analyze',
            r'evaluate', r'assess', r'examine', r'investigate',
            r'compare\s+.*\s+with', r'relationship\s+between',
            r'implications\s+of', r'impact\s+of'
        ]
        
        # Check for comparative intent
        comparative_patterns = [
            r'compare', r'contrast', r'difference\s+between', r'versus',
            r'better', r'worse', r'advantages\s+of', r'disadvantages\s+of',
            r'pros\s+and\s+cons', r'similarities\s+between'
        ]
        
        # Check for procedural intent
        procedural_patterns = [
            r'how\s+to', r'steps\s+to', r'process\s+of', r'procedure\s+for',
            r'guide\s+for', r'instructions\s+for', r'method\s+of',
            r'way\s+to', r'create', r'implement', r'develop', r'build'
        ]
        
        # Check for exploratory intent
        exploratory_patterns = [
            r'explore', r'discover', r'possibilities', r'alternatives',
            r'options\s+for', r'approaches\s+to', r'ways\s+to',
            r'tell\s+me\s+more', r'what\s+else', r'other\s+examples'
        ]
        
        # Check for clarification intent
        clarification_patterns = [
            r'clarify', r'explain\s+again', r'what\s+do\s+you\s+mean',
            r'don\'t\s+understand', r'confused\s+about', r'elaborate\s+on',
            r'clearer\s+explanation', r'more\s+details', r'can\s+you\s+rephrase'
        ]
        
        # Check each pattern group
        for patterns, intent in [
            (info_patterns, QueryIntent.INFORMATIONAL),
            (analytical_patterns, QueryIntent.ANALYTICAL),
            (comparative_patterns, QueryIntent.COMPARATIVE),
            (procedural_patterns, QueryIntent.PROCEDURAL),
            (exploratory_patterns, QueryIntent.EXPLORATORY),
            (clarification_patterns, QueryIntent.CLARIFICATION)]:
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        # Default to informational if no clear intent is detected
        return QueryIntent.INFORMATIONAL
    
    def _enhance_query(
        self, 
        query: str,
        conversation_context: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Enhance or reformulate an ambiguous query
        
        Args:
            query: Original query string
            conversation_context: Previous conversation turns
            
        Returns:
            Enhanced query string
        """
        if not self.llm_query_fn:
            return query
            
        # Simplified enhancement for ambiguous queries
        enhancement_prompt = f"""
        Original query: "{query}"
        
        Please reformulate this query to be more specific and detailed, while preserving the original intent.
        The reformulated query should be clearer and easier to answer precisely.
        Return ONLY the reformulated query without any explanations.
        """
        
        try:
            # Call the LLM query function
            enhanced_query = self.llm_query_fn(enhancement_prompt)
            
            # Clean up the response (remove quotes, etc.)
            enhanced_query = enhanced_query.strip()
            enhanced_query = re.sub(r'^["\'](.*)["\']\Z', r'\1', enhanced_query)
            
            return enhanced_query if enhanced_query else query
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return query
    
    def _handle_structured_query(
        self,
        query: str,
        classification: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle structured (SQL-like) queries
        
        Args:
            query: Original query string
            classification: Query classification information
            metadata: Optional query metadata
            
        Returns:
            Routing decision and query plan
        """
        logger.info(f"Handling structured query: {query}")
        
        if not self.sql_query_fn:
            logger.warning("SQL query function not provided, falling back to LLM")
            return self._fallback_routing_strategy(query, classification, None, metadata)
        
        # Extract any table/database information from metadata if available
        db_info = {}
        if metadata and "database_info" in metadata:
            db_info = metadata["database_info"]
        
        # Prepare query plan
        query_plan = {
            "query_type": classification["query_type"].value,
            "handler": "sql_query_generator",
            "parameters": {
                "original_query": query,
                "temperature": self.config["sql"]["generation_temperature"],
                "max_tokens": self.config["sql"]["max_tokens"],
                "safety_checks": self.config["sql"]["safety_checks"],
                "database_info": db_info
            },
            "execution_plan": {
                "generate_sql": True,
                "validate_sql": self.config["sql"]["safety_checks"],
                "execute_sql": True,
                "format_results": True
            }
        }
        
        return {
            "success": True,
            "query_info": classification,
            "query_plan": query_plan
        }
    
    def _handle_unstructured_query(
        self,
        query: str,
        classification: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle unstructured text queries
        
        Args:
            query: Original query string
            classification: Query classification information
            metadata: Optional query metadata
            
        Returns:
            Routing decision and query plan
        """
        logger.info(f"Handling unstructured query: {query}")
        
        # Determine if we should use vector search, keyword search, or hybrid
        search_strategy = "hybrid"
        if metadata and "preferred_search" in metadata:
            search_strategy = metadata["preferred_search"]
        
        # Configure search parameters
        top_k = self.config["retrieval"]["top_k"]
        if metadata and "top_k" in metadata:
            top_k = metadata["top_k"]
        
        # Prepare query plan
        query_plan = {
            "query_type": classification["query_type"].value,
            "handler": "retrieval_search",
            "parameters": {
                "original_query": query,
                "search_strategy": search_strategy,
                "vector_weight": self.config["retrieval"]["vector_weight"],
                "keyword_weight": self.config["retrieval"]["keyword_weight"],
                "top_k": top_k,
                "diversity_factor": self.config["retrieval"]["diversity_factor"]
            },
            "execution_plan": {
                "vector_search": search_strategy in ["vector", "hybrid"],
                "keyword_search": search_strategy in ["keyword", "hybrid"],
                "rerank_results": search_strategy == "hybrid",
                "format_context": True
            }
        }
        
        return {
            "success": True,
            "query_info": classification,
            "query_plan": query_plan
        }
    
    def _handle_hybrid_query(
        self,
        query: str,
        classification: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle hybrid queries requiring both structured and unstructured data
        
        Args:
            query: Original query string
            classification: Query classification information
            metadata: Optional query metadata
            
        Returns:
            Routing decision and query plan
        """
        logger.info(f"Handling hybrid query: {query}")
        
        # For hybrid queries, we'll need both SQL and retrieval capabilities
        if not self.sql_query_fn or not self.vector_search_fn:
            logger.warning("Missing required functions for hybrid query, using available methods")
        
        # Configure search parameters
        top_k = max(2, self.config["retrieval"]["top_k"] - 2)  # Reduce for hybrid to make room for SQL results
        
        # Prepare query plan with both SQL and retrieval components
        query_plan = {
            "query_type": classification["query_type"].value,
            "handler": "hybrid_search",
            "parameters": {
                "original_query": query,
                "sql_parameters": {
                    "temperature": self.config["sql"]["generation_temperature"],
                    "safety_checks": self.config["sql"]["safety_checks"]
                },
                "retrieval_parameters": {
                    "search_strategy": "hybrid",
                    "vector_weight": self.config["retrieval"]["vector_weight"],
                    "keyword_weight": self.config["retrieval"]["keyword_weight"],
                    "top_k": top_k
                }
            },
            "execution_plan": {
                "decompose_query": True,
                "sql_query": self.sql_query_fn is not None,
                "retrieval_search": self.vector_search_fn is not None or self.keyword_search_fn is not None,
                "combine_results": True
            }
        }
        
        return {
            "success": True,
            "query_info": classification,
            "query_plan": query_plan
        }
    
    def _handle_conversational_query(
        self,
        query: str,
        classification: Dict[str, Any],
        conversation_context: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle conversational/follow-up queries
        
        Args:
            query: Original query string
            classification: Query classification information
            conversation_context: Previous conversation turns
            metadata: Optional query metadata
            
        Returns:
            Routing decision and query plan
        """
        logger.info(f"Handling conversational query: {query}")
        
        # Get original query type if available
        original_type = classification.get("original_query_type", QueryType.UNKNOWN)
        
        # If we couldn't determine the original type, default to unstructured
        if original_type == QueryType.UNKNOWN:
            original_type = QueryType.UNSTRUCTURED
        
        # Prepare query plan based on the original query type
        query_plan = {
            "query_type": classification["query_type"].value,
            "original_type": original_type.value,
            "handler": "conversational_handler",
            "parameters": {
                "original_query": query,
                "conversation_history": conversation_context,
                "is_followup": classification.get("is_followup", True)
            },
            "execution_plan": {
                "resolve_references": True,
                "reconstruct_full_query": True,
                "execute_based_on_original_type": True
            }
        }
        
        return {
            "success": True,
            "query_info": classification,
            "query_plan": query_plan
        }
    
    def _handle_calculation_query(
        self,
        query: str,
        classification: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle calculation/mathematical queries
        
        Args:
            query: Original query string
            classification: Query classification information
            metadata: Optional query metadata
            
        Returns:
            Routing decision and query plan
        """
        logger.info(f"Handling calculation query: {query}")
        
        # Prepare query plan for calculation
        query_plan = {
            "query_type": classification["query_type"].value,
            "handler": "calculation_handler",
            "parameters": {
                "original_query": query,
                "precision": metadata.get("precision", 4) if metadata else 4,
                "show_work": metadata.get("show_work", True) if metadata else True
            },
            "execution_plan": {
                "extract_calculation": True,
                "validate_formula": True,
                "compute_result": True,
                "show_steps": metadata.get("show_work", True) if metadata else True
            }
        }
        
        return {
            "success": True,
            "query_info": classification,
            "query_plan": query_plan
        }
    
    def _fallback_routing_strategy(
        self,
        query: str,
        classification: Dict[str, Any],
        conversation_context: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Implement fallback strategy when classification is uncertain
        
        Args:
            query: Original query string
            classification: Query classification information
            conversation_context: Previous conversation turns
            metadata: Optional query metadata
            
        Returns:
            Routing decision with fallback plan
        """
        logger.info(f"Using fallback strategy for query: {query}")
        
        # 1. Try to enhance/reformulate the query if enabled
        enhanced_query = query
        if self.config["enable_query_enhancement"] and self.llm_query_fn:
            try:
                enhanced_query = self._enhance_query(query, conversation_context)
                logger.info(f"Enhanced query: {enhanced_query}")
            except Exception as e:
                logger.error(f"Error enhancing query: {e}")
        
        # 2. Increase retrieval count for better coverage
        top_k = self.config["default_top_k"] * self.config["fallback_retrieval_increase"]
        
        # 3. Use hybrid search with balanced weights
        fallback_plan = {
            "query_type": "fallback",
            "handler": "hybrid_retrieval",
            "parameters": {
                "original_query": query,
                "enhanced_query": enhanced_query,
                "search_strategy": "hybrid",
                "vector_weight": 0.5,  # Balanced weights for fallback
                "keyword_weight": 0.5,
                "top_k": top_k,
                "diversity_factor": self.config["diversity_factor"] * 1.5  # Increase diversity
            },
            "execution_plan": {
                "use_enhanced_query": enhanced_query != query,
                "vector_search": True,
                "keyword_search": True,
                "use_larger_context": True,
                "format_with_high_precision": True
            }
        }
        
        return {
            "success": True,
            "query_info": classification,
            "is_fallback": True,
            "query_plan": fallback_plan,
            "original_classification": classification
        }

def test_query_router():
    """Test the query router with sample queries."""
    
    # Define mock search/query functions
    def mock_vector_search(query, top_k=5):
        return [{"id": i, "score": 0.9 - (i * 0.1), "text": f"Vector result {i}"} for i in range(top_k)]
    
    def mock_keyword_search(query, top_k=5):
        return [{"id": i, "score": 0.8 - (i * 0.1), "text": f"Keyword result {i}"} for i in range(top_k)]
    
    def mock_sql_query(query):
        return {"columns": ["id", "value"], "rows": [[1, 100], [2, 200]]}
    
    def mock_llm_query(prompt):
        return f"This is a response to: {prompt[:50]}..."
    
    # Initialize query router with mock functions
    router = QueryRouter(
        vector_search_fn=mock_vector_search,
        keyword_search_fn=mock_keyword_search,
        sql_query_fn=mock_sql_query,
        llm_query_fn=mock_llm_query
    )
    
    # Test queries for different types
    test_queries = {
        "structured": "Show me the total sales by region for Q2 2023",
        "unstructured": "Explain how retrieval augmented generation works and its benefits",
        "hybrid": "Compare our sales data with the market analysis in the quarterly report",
        "conversational": "Tell me more about that",
        "calculation": "Calculate the ROI for a $10,000 investment with 7% annual return over 5 years",
        "ambiguous": "Check the data"
    }
    
    # Sample conversation context for testing conversational queries
    sample_context = [
        {
            "role": "user",
            "content": "Explain how vector databases work",
            "query_type": "unstructured"
        },
        {
            "role": "assistant",
            "content": "Vector databases store and index high-dimensional vectors..."
        }
    ]
    
    results = {}
    
    # Test each query type
    for query_name, query_text in test_queries.items():
        print(f"\nTesting {query_name} query: '{query_text}'")
        
        # For conversational query, include conversation context
        if query_name == "conversational":
            result = router.route_query(query_text, conversation_context=sample_context)
        else:
            result = router.route_query(query_text)
            
        # Store result
        results[query_name] = result
        
        # Print key information
        print(f"Classified as: {result.get('query_info', {}).get('query_type', 'Unknown')}")
        print(f"Confidence: {result.get('query_info', {}).get('confidence', 0):.2f}")
        print(f"Handler: {result.get('query_plan', {}).get('handler', 'Unknown')}")
    
    return results

# Run the test when executed as a script
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("QUERY ROUTING SYSTEM DEMONSTRATION")
    print("=" *50)
    test_query_router()