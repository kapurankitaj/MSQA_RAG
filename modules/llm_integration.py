import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

# LLM and API Integration
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Supporting libraries
import tiktoken
import uuid
import hashlib
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage


# ============= CONFIGURATION SETTINGS =============
# Configuration dictionary
CONFIG = {
    # Logging Configuration
    "logging": {
        "level": logging.INFO,
        "format": '%(asctime)s - %(levelname)s: %(message)s',
        "handlers": [
            "file_handler",
            "stream_handler"
        ],
        "log_file": "llm_integration.log"
    },
    
    # LLM and API Settings
    "llm": {
        "api_key_env": "GROQ_API_KEY",
        "default_model": "llama3-8b-8192",
        "temperature": 0.3
    },
    
    # Token Settings
    "tokens": {
        "max_context_tokens": 6000,
        "response_token_limit": 1000
    },
    
    # Query Type Classification
    "query_types": {
        "factual": ["what", "who", "when", "where"],
        "explanatory": ["why", "how"],
        "comparative": ["compare", "contrast"],
        "analytical": ["analyze", "breakdown", "evaluate"],
        "structured_data": ["count", "sum", "average", "total"]
    },
    
    # Citation Styles
    "citation_styles": ["inline", "footnote", "end_references"]
}

# Configure logging
logging.basicConfig(
    level=CONFIG["logging"]["level"],
    format=CONFIG["logging"]["format"],
    handlers=[
        logging.FileHandler(CONFIG["logging"]["log_file"]),
        logging.StreamHandler()
    ]
)

class LLMIntegrationSystem:
    def __init__(
        self, 
        model=None, 
        max_context_tokens=None,
        response_token_limit=None,
        config=CONFIG
    ):
        """
        Initialize LLM Integration System
        
        Args:
            model (str): Groq LLM model to use
            max_context_tokens (int): Maximum tokens for context window
            response_token_limit (int): Maximum tokens for generated response
            config (dict): Configuration dictionary
        """
        # Store configuration
        self.config = config
        
        # Use parameters or defaults from config
        self.model = model or config["llm"]["default_model"]
        self.max_context_tokens = max_context_tokens or config["tokens"]["max_context_tokens"]
        self.response_token_limit = response_token_limit or config["tokens"]["response_token_limit"]
        
        # Load environment variables
        load_dotenv()
        
        # Initialize Groq LLM
        self.llm = self._initialize_llm(self.model)
        
        # Tokenizer for managing context window
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Conversation memory - using newer approach
        from langchain_community.chat_message_histories import ChatMessageHistory
        self.chat_history = ChatMessageHistory()
        
        # Prompt templates
        self._initialize_prompt_templates()
        
        # Citation tracking
        self.citation_tracker = CitationManager(config)
    
    def _initialize_llm(self, model):
        """Initialize Groq LLM with error handling"""
        try:
            api_key = os.getenv(self.config["llm"]["api_key_env"])
            if not api_key:
                raise ValueError(f"API key not found in environment variable {self.config['llm']['api_key_env']}")
            
            llm = ChatGroq(
                model=model, 
                api_key=api_key, 
                temperature=self.config["llm"]["temperature"]
            )
            
            logging.info(f"Initialized Groq LLM with model: {model}")
            return llm
        except Exception as e:
            logging.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _initialize_prompt_templates(self):
        """Create prompt templates for different query types"""
        self.prompt_templates = {
            'default': PromptTemplate(
                input_variables=['context', 'question', 'chat_history'],
                template="""You are a helpful AI assistant providing comprehensive and accurate responses.

Context Information:
{context}

Chat History:
{chat_history}

Current Question: {question}

Please provide a detailed, well-structured response based on the given context. If the context does not contain sufficient information, clearly state that you cannot fully answer the question with the available information."""
            ),
            'factual': PromptTemplate(
                input_variables=['context', 'question', 'chat_history'],
                template="""Focus on providing precise, fact-based information.

Context Information:
{context}

Chat History:
{chat_history}

Current Question: {question}

Provide a concise, direct answer with key facts extracted from the context."""
            ),
            'explanatory': PromptTemplate(
                input_variables=['context', 'question', 'chat_history'],
                template="""Provide a comprehensive explanation with clear reasoning.

Context Information:
{context}

Chat History:
{chat_history}

Current Question: {question}

Break down the explanation into clear, logical steps. Use analogies or examples if helpful."""
            )
        }
    
    def classify_query_type(self, query: str) -> str:
        """
        Classify the type of query to select appropriate prompt template
        
        Args:
            query (str): Input query to classify
        
        Returns:
            str: Classified query type
        """
        # Convert query to lowercase for easier matching
        query_lower = query.lower().strip()
        
        # First word-based classification
        first_word = query_lower.split()[0] if query_lower else ''
        
        # Check query type based on first word
        for query_type, keywords in self.config["query_types"].items():
            if first_word in keywords:
                return query_type
        
        # Default to general query type
        return 'default'
    
    def truncate_context(self, context_docs: List[Dict], max_tokens: int) -> List[Dict]:
        """
        Truncate context documents to fit within token limit
        
        Args:
            context_docs (List[Dict]): List of context documents
            max_tokens (int): Maximum number of tokens allowed
        
        Returns:
            List[Dict]: Truncated context documents
        """
        truncated_docs = []
        current_tokens = 0
        
        for doc in context_docs:
            # Estimate tokens for this document
            doc_text = doc.get('text', '')
            doc_tokens = len(self.tokenizer.encode(doc_text))
            
            # Add document if it fits within remaining token budget
            if current_tokens + doc_tokens <= max_tokens:
                truncated_docs.append(doc)
                current_tokens += doc_tokens
            else:
                break
        
        return truncated_docs
    
    def generate_response(
        self, 
        query: str, 
        context_docs: List[Dict], 
        chat_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using retrieved context and LLM
        
        Args:
            query (str): User's input query
            context_docs (List[Dict]): Retrieved context documents
            chat_history (Optional[List]): Previous conversation context
        
        Returns:
            Dict[str, Any]: Generated response with metadata
        """
        try:
            # Classify query type
            query_type = self.classify_query_type(query)
            
            # Truncate context to fit token limit
            truncated_context = self.truncate_context(context_docs, self.max_context_tokens)
            
            # Check if context is empty or invalid
            if not truncated_context or not any(doc.get('text') for doc in truncated_context):
                context_str = "No relevant information found in the available documents."
            else:
                # Prepare context string with clear document separation
                context_str = "\n\n".join([
                    f"[Source {i+1}] {doc.get('text', '')}" 
                    for i, doc in enumerate(truncated_context)
                    if doc.get('text')  # Only include documents with text
                ])
            
            # Prepare chat history
            history_str = ""
            if chat_history:
                if isinstance(chat_history, str):
                    history_str = chat_history
                elif isinstance(chat_history, list):
                    history_str = "\n".join([f"{msg.get('role', 'User')}: {msg.get('content', '')}" for msg in chat_history])
            
            # Create system prompt
            system_prompt = """You are a helpful AI assistant providing accurate responses based on the given context.
    If the context contains relevant information, use it in your response.
    If the information is incomplete, acknowledge this and explain what you can based on the available context.
    IMPORTANT: Use numbered citations [1], [2], etc. to reference specific sources in your response."""
            
            # Create messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""Context Information:
    {context_str}

    Chat History:
    {history_str}

    Current Question: {query}

    Please provide a detailed response using the provided context information.""")
            ]
            
            # Generate response
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # Process citations from the context docs
            citations = self.citation_tracker.process_sources(truncated_context)
            
            # Prepare response metadata
            response_metadata = {
                'query': query,
                'query_type': query_type,
                'context_docs': truncated_context,
                'citations': citations,
                'response_id': str(uuid.uuid4())
            }
            
            logging.info(f"Generated response for query: {query}")
            
            return {
                'response': response_text,
                'metadata': response_metadata
            }
        
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return {
                'response': "I apologize, but I couldn't generate a complete response.",
                'metadata': {
                    'error': str(e)
                }
            }

class CitationManager:
    """
    Manages citation generation and tracking
    """
    
    def process_sources(self, context_docs: List[Dict], style: str = 'inline') -> Dict[str, Any]:
        """
        Process source documents to create citation objects
        
        Args:
            context_docs (List[Dict]): Context documents
            style (str): Citation style
            
        Returns:
            Dict[str, Any]: Processed citations
        """
        citations = []
        for i, doc in enumerate(context_docs, 1):
            metadata = doc.get('metadata', {})
            doc_type = metadata.get('document_type', 'unknown')
            
            # Create citation object
            citation = {
                'id': hashlib.md5(str(metadata).encode()).hexdigest()[:12],
                'index': i,
                'marker': f"[{i}]",
                'text': f"{i}. {metadata.get('title', metadata.get('source', f'Source {i}'))}, {doc_type}",
                'source_info': metadata,
                'timestamp': datetime.now().isoformat()
            }
            citations.append(citation)
        
        return {
            'citations': citations,
            'style': style,
            'unique_count': len(set(c['id'] for c in citations))
        }
    
    def __init__(self, config=CONFIG):
        """Initialize citation tracking mechanism"""
        self.config = config
        self.citation_styles = {
            'inline': self._generate_inline_citations,
            'footnote': self._generate_footnote_citations,
            'end_references': self._generate_end_references
        }
    
    def generate_citations(
        self, 
        context_docs: List[Dict], 
        style: str = 'inline'
    ) -> Dict[str, Any]:
        """
        Generate citations for context documents
        
        Args:
            context_docs (List[Dict]): Context documents
            style (str): Citation style (inline, footnote, end_references)
        
        Returns:
            Dict[str, Any]: Generated citations
        """
        try:
            # Select citation style
            citation_func = self.citation_styles.get(
                style, 
                self.citation_styles['inline']
            )
            
            # Generate citations
            return citation_func(context_docs)
        
        except Exception as e:
            logging.error(f"Error generating citations: {e}")
            return {}
    
    def _generate_inline_citations(self, context_docs: List[Dict]) -> Dict[str, Any]:
        """
        Generate inline citations
        """
        citations = []
        for i, doc in enumerate(context_docs, 1):
            source = doc.get('metadata', {}).get('source', f'Source {i}')
            citations.append({
                'index': i,
                'source': source,
                'inline_marker': f'[{i}]'
            })
        
        return {
            'style': 'inline',
            'citations': citations
        }
    
    def _generate_footnote_citations(self, context_docs: List[Dict]) -> Dict[str, Any]:
        """
        Generate footnote-style citations
        """
        citations = []
        for i, doc in enumerate(context_docs, 1):
            source = doc.get('metadata', {}).get('source', f'Source {i}')
            citations.append({
                'index': i,
                'source': source,
                'footnote_text': f'[{i}] {source}'
            })
        
        return {
            'style': 'footnote',
            'citations': citations
        }
    
    def _generate_end_references(self, context_docs: List[Dict]) -> Dict[str, Any]:
        """
        Generate end references
        """
        references = []
        for i, doc in enumerate(context_docs, 1):
            metadata = doc.get('metadata', {})
            references.append({
                'index': i,
                'source': metadata.get('source', f'Source {i}'),
                'details': {
                    'title': metadata.get('title', 'Untitled'),
                    'document_type': metadata.get('document_type', 'Unknown'),
                    'source_type': metadata.get('source_name', 'Unknown')
                }
            })
        
        return {
            'style': 'end_references',
            'references': references
        }

def main():
    """
    Demonstrate LLM Integration System functionality
    """
    try:
        # Verbose logging
        logging.info("Starting LLM Integration System Demo")

        # Load environment variables first
        load_dotenv()

        # Check API key
        api_key = os.getenv(CONFIG["llm"]["api_key_env"])
        if not api_key:
            raise ValueError(f"{CONFIG['llm']['api_key_env']} not found in environment variables")
        
        logging.info("API key successfully loaded")

        # Initialize LLM Integration System
        try:
            llm_system = LLMIntegrationSystem()
        except Exception as init_error:
            logging.error(f"Failed to initialize LLM system: {init_error}")
            raise
        
        # Example context documents (simulated retrieval)
        example_context = [
            {
                'text': "Retrieval Augmented Generation (RAG) is an advanced AI technique that enhances large language models by incorporating external knowledge during response generation.",
                'metadata': {
                    'source': 'AI Techniques Handbook',
                    'document_type': 'technical_article'
                }
            },
            {
                'text': "RAG improves the accuracy and relevance of AI-generated responses by dynamically retrieving and integrating contextual information from a knowledge base.",
                'metadata': {
                    'source': 'Machine Learning Research Paper',
                    'document_type': 'research_paper'
                }
            }
        ]
        
        # Example queries
        queries = [
            "What is Retrieval Augmented Generation?",
            "How does RAG improve AI responses?"
        ]
        
        # Process each query
        for query in queries:
            print(f"\nQuery: {query}")
            print("=" * 50)
            
            try:
                # Generate response
                response_data = llm_system.generate_response(
                    query, 
                    context_docs=example_context
                )
                
                # Print response
                print("\nResponse:")
                print(response_data.get('response', 'No response generated'))
                
                # Print citations
                print("\nCitations:")
                print(json.dumps(
                    response_data.get('metadata', {}).get('citations', {}), 
                    indent=2
                ))
                
                logging.info(f"Successfully processed query: {query}")
            
            except Exception as query_error:
                logging.error(f"Error processing query '{query}': {query_error}")
                print(f"Error processing query: {query_error}")
    
    except Exception as e:
        logging.error(f"Critical error in main execution: {e}")
        print(f"A critical error occurred: {e}")
        import traceback
        traceback.print_exc()

# Ensure proper script execution
if __name__ == "__main__":
    main()