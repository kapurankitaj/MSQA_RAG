"""
Advanced features module for enhancing RAG system capabilities.
This module implements query rewriting, source reliability scoring,
personalization options, and conversational context preservation.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename='advanced_features.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('advanced_features')

class QueryRewriter:
    """
    Implements query rewriting to improve retrieval performance.
    """
    
    def __init__(self, llm_service):
        """
        Initialize QueryRewriter with an LLM service.
        
        Args:
            llm_service: A reference to the LLM integration module
        """
        self.llm_service = llm_service
        self.rewrite_history = {}
        self.rewrite_cache = {}
        logger.info("QueryRewriter initialized")
        
    def rewrite_query(self, original_query: str, context: Optional[List[Dict]] = None) -> str:
        """
        Rewrite a query to improve retrieval performance.
        
        Args:
            original_query: The original user query
            context: Optional conversation context
            
        Returns:
            str: The rewritten query
        """
        # Check cache first
        if original_query in self.rewrite_cache:
            logger.info(f"Using cached rewrite for: {original_query}")
            return self.rewrite_cache[original_query]
        
        # Prepare system prompt for query rewriting
        system_prompt = """
        Your task is to rewrite a search query to make it more effective for retrieval.
        Consider:
        1. Adding specific keywords that would appear in relevant documents
        2. Expanding abbreviations
        3. Including synonyms for key terms
        4. Removing unnecessary words
        5. Making implicit information explicit
        
        Return only the rewritten query without explanations.
        """
        
        # Include conversation context if available
        context_str = ""
        if context:
            recent_exchanges = context[-3:] if len(context) > 3 else context
            context_str = "Previous conversation:\n" + "\n".join([
                f"User: {exchange.get('user', '')}\nAssistant: {exchange.get('assistant', '')}"
                for exchange in recent_exchanges
            ])
        
        # Construct the prompt
        prompt = f"{context_str}\n\nOriginal query: {original_query}\n\nRewritten query:"
        
        try:
            # Get rewritten query from LLM
            rewritten_query = self.llm_service.generate_text(system_prompt, prompt, max_tokens=100).strip()
            
            # Log and store the rewrite
            logger.info(f"Original: '{original_query}' -> Rewritten: '{rewritten_query}'")
            self.rewrite_history[original_query] = rewritten_query
            self.rewrite_cache[original_query] = rewritten_query
            
            # Save history periodically (could be optimized)
            if len(self.rewrite_history) % 10 == 0:
                self._save_history()
                
            return rewritten_query
        
        except Exception as e:
            logger.error(f"Query rewriting failed: {str(e)}")
            return original_query
    
    def _save_history(self):
        """Save query rewrite history to disk"""
        try:
            os.makedirs('data/advanced', exist_ok=True)
            with open('data/advanced/query_rewrites.json', 'w') as f:
                json.dump(self.rewrite_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save query rewrite history: {str(e)}")


class SourceReliabilityScorer:
    """
    Implements source reliability scoring for retrieved documents.
    """
    
    def __init__(self):
        """Initialize the source reliability scorer"""
        self.reliability_scores = {}
        self._load_reliability_data()
        logger.info("SourceReliabilityScorer initialized")
        
    def _load_reliability_data(self):
        """Load saved reliability data if available"""
        try:
            if os.path.exists('data/advanced/source_reliability.json'):
                with open('data/advanced/source_reliability.json', 'r') as f:
                    self.reliability_scores = json.load(f)
                logger.info(f"Loaded {len(self.reliability_scores)} source reliability scores")
        except Exception as e:
            logger.error(f"Failed to load reliability data: {str(e)}")
            
    def _save_reliability_data(self):
        """Save reliability data to disk"""
        try:
            os.makedirs('data/advanced', exist_ok=True)
            with open('data/advanced/source_reliability.json', 'w') as f:
                json.dump(self.reliability_scores, f, indent=2)
            logger.info(f"Saved {len(self.reliability_scores)} source reliability scores")
        except Exception as e:
            logger.error(f"Failed to save reliability data: {str(e)}")
    
    def score_source(self, source_id: str, metadata: Dict) -> float:
        """
        Score the reliability of a source based on various factors.
        
        Args:
            source_id: Unique identifier for the source
            metadata: Metadata for the source
            
        Returns:
            float: Reliability score between 0 and 1
        """
        # Return cached score if available
        if source_id in self.reliability_scores:
            return self.reliability_scores[source_id]
        
        # Calculate base score
        base_score = 0.5  # Default neutral score
        
        # Factor 1: Source type
        source_type = metadata.get('type', '').lower()
        if source_type in ['pdf', 'academic', 'peer_reviewed']:
            base_score += 0.2
        elif source_type in ['html', 'webpage']:
            base_score += 0.0  # Neutral
        elif source_type in ['social_media', 'forum']:
            base_score -= 0.1
            
        # Factor 2: Author information
        if metadata.get('author'):
            base_score += 0.1
            
        # Factor 3: Publication date
        if metadata.get('date'):
            # Prefer more recent sources for most topics
            # This is a simplified implementation
            base_score += 0.05
            
        # Factor 4: Domain reputation (simplified)
        domain = metadata.get('domain', '')
        academic_domains = ['.edu', '.gov', '.org']
        if any(domain.endswith(d) for d in academic_domains):
            base_score += 0.1
            
        # Ensure score is between 0 and 1
        final_score = max(0.0, min(1.0, base_score))
        
        # Cache the score
        self.reliability_scores[source_id] = final_score
        
        # Save periodically
        if len(self.reliability_scores) % 10 == 0:
            self._save_reliability_data()
            
        return final_score
        
    def adjust_search_results(self, results: List[Dict]) -> List[Dict]:
        """
        Adjust search results based on source reliability.
        
        Args:
            results: List of search results with scores
            
        Returns:
            List[Dict]: Adjusted search results
        """
        if not results:
            return results
            
        # Score each result
        for result in results:
            source_id = result.get('id', '')
            metadata = result.get('metadata', {})
            reliability = self.score_source(source_id, metadata)
            
            # Adjust the original score
            original_score = result.get('score', 0.5)
            # Weighted combination (70% original score, 30% reliability)
            adjusted_score = (0.7 * original_score) + (0.3 * reliability)
            
            # Update the result
            result['original_score'] = original_score
            result['reliability_score'] = reliability
            result['score'] = adjusted_score
            
        # Re-sort based on adjusted scores
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return results


class PersonalizationManager:
    """
    Implements personalization options for the RAG system.
    """
    
    def __init__(self):
        """Initialize personalization manager"""
        self.user_profiles = {}
        self._load_profiles()
        logger.info("PersonalizationManager initialized")
        
    def _load_profiles(self):
        """Load user profiles from disk"""
        try:
            if os.path.exists('data/advanced/user_profiles.json'):
                with open('data/advanced/user_profiles.json', 'r') as f:
                    self.user_profiles = json.load(f)
                logger.info(f"Loaded {len(self.user_profiles)} user profiles")
        except Exception as e:
            logger.error(f"Failed to load user profiles: {str(e)}")
    
    def _save_profiles(self):
        """Save user profiles to disk"""
        try:
            os.makedirs('data/advanced', exist_ok=True)
            with open('data/advanced/user_profiles.json', 'w') as f:
                json.dump(self.user_profiles, f, indent=2)
            logger.info(f"Saved {len(self.user_profiles)} user profiles")
        except Exception as e:
            logger.error(f"Failed to save user profiles: {str(e)}")
    
    def get_or_create_profile(self, user_id: str) -> Dict:
        """
        Get a user profile or create a new one if it doesn't exist.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Dict: User profile
        """
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'preferences': {
                    'detail_level': 'medium',  # ['low', 'medium', 'high']
                    'citation_style': 'inline',  # ['inline', 'footnote', 'endnote']
                    'technical_level': 'medium',  # ['beginner', 'medium', 'expert']
                },
                'history': {
                    'topics': {},  # Topic -> frequency
                    'documents': {},  # Document ID -> access count
                    'feedback': {},  # Response ID -> feedback score
                }
            }
            self._save_profiles()
            
        return self.user_profiles[user_id]
    
    def update_profile(self, user_id: str, profile_updates: Dict) -> Dict:
        """
        Update a user profile with new information.
        
        Args:
            user_id: Unique user identifier
            profile_updates: Dictionary of updates to apply
            
        Returns:
            Dict: Updated user profile
        """
        profile = self.get_or_create_profile(user_id)
        
        # Update preferences
        if 'preferences' in profile_updates:
            for k, v in profile_updates['preferences'].items():
                profile['preferences'][k] = v
                
        # Update history
        if 'history' in profile_updates:
            for category, updates in profile_updates['history'].items():
                if category not in profile['history']:
                    profile['history'][category] = {}
                    
                for k, v in updates.items():
                    if k in profile['history'][category]:
                        # Increment existing values
                        profile['history'][category][k] += v
                    else:
                        # Add new values
                        profile['history'][category][k] = v
        
        # Save changes
        self._save_profiles()
        return profile
    
    def personalize_query(self, user_id: str, query: str) -> str:
        """
        Personalize a query based on user profile.
        
        Args:
            user_id: Unique user identifier
            query: Original query
            
        Returns:
            str: Personalized query
        """
        profile = self.get_or_create_profile(user_id)
        
        # Simple personalization example:
        # Add frequent topics as context for better results
        topics = profile['history'].get('topics', {})
        if topics:
            # Get top 3 most frequent topics
            top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]
            top_topics = [topic for topic, _ in top_topics]
            
            # Add topics as context
            personalized_query = f"{query} (Context: {', '.join(top_topics)})"
            logger.info(f"Personalized query for {user_id}: {personalized_query}")
            return personalized_query
            
        return query
    
    def personalize_results(self, user_id: str, results: List[Dict]) -> List[Dict]:
        """
        Personalize search results based on user profile.
        
        Args:
            user_id: Unique user identifier
            results: List of search results
            
        Returns:
            List[Dict]: Personalized results
        """
        profile = self.get_or_create_profile(user_id)
        
        # Update document access counts
        for result in results:
            doc_id = result.get('id', '')
            if doc_id:
                if 'documents' not in profile['history']:
                    profile['history']['documents'] = {}
                    
                profile['history']['documents'][doc_id] = profile['history']['documents'].get(doc_id, 0) + 1
        
        # Prioritize results based on user's technical level
        tech_level = profile['preferences'].get('technical_level', 'medium')
        
        # Adjust scores based on technical level
        for result in results:
            original_score = result.get('score', 0.5)
            result_tech_level = result.get('metadata', {}).get('technical_level', 'medium')
            
            # Boost score if technical level matches user preference
            if result_tech_level == tech_level:
                result['score'] = min(1.0, original_score * 1.2)  # 20% boost
                
        # Re-sort results
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Save updated profile
        self._save_profiles()
        
        return results


class ConversationManager:
    """
    Manages conversation history and context preservation.
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation manager.
        
        Args:
            max_history: Maximum number of exchanges to keep in history
        """
        self.conversations = {}  # user_id -> conversation history
        self.max_history = max_history
        self._load_conversations()
        logger.info(f"ConversationManager initialized with max_history={max_history}")
        
    def _load_conversations(self):
        """Load conversation histories from disk"""
        try:
            if os.path.exists('data/advanced/conversations.json'):
                with open('data/advanced/conversations.json', 'r') as f:
                    self.conversations = json.load(f)
                logger.info(f"Loaded {len(self.conversations)} conversations")
        except Exception as e:
            logger.error(f"Failed to load conversations: {str(e)}")
    
    def _save_conversations(self):
        """Save conversation histories to disk"""
        try:
            os.makedirs('data/advanced', exist_ok=True)
            with open('data/advanced/conversations.json', 'w') as f:
                json.dump(self.conversations, f, indent=2)
            logger.info(f"Saved {len(self.conversations)} conversations")
        except Exception as e:
            logger.error(f"Failed to save conversations: {str(e)}")
    
    def add_exchange(self, user_id: str, query: str, response: str, 
                     context_docs: Optional[List[Dict]] = None) -> None:
        """
        Add a new exchange to the conversation history.
        
        Args:
            user_id: Unique user identifier
            query: User query
            response: System response
            context_docs: Optional list of documents used for context
        """
        if user_id not in self.conversations:
            self.conversations[user_id] = []
            
        # Create exchange record
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'user_query': query,
            'system_response': response,
            'context_docs': context_docs if context_docs else []
        }
        
        # Add to history and maintain max size
        self.conversations[user_id].append(exchange)
        if len(self.conversations[user_id]) > self.max_history:
            self.conversations[user_id] = self.conversations[user_id][-self.max_history:]
            
        # Save periodically
        if sum(len(conv) for conv in self.conversations.values()) % 10 == 0:
            self._save_conversations()
    
    def get_conversation_context(self, user_id: str, max_exchanges: Optional[int] = None) -> List[Dict]:
        """
        Get conversation history for a user.
        
        Args:
            user_id: Unique user identifier
            max_exchanges: Optional maximum number of exchanges to retrieve
            
        Returns:
            List[Dict]: Conversation history
        """
        if user_id not in self.conversations:
            return []
            
        history = self.conversations[user_id]
        if max_exchanges and max_exchanges > 0:
            history = history[-max_exchanges:]
            
        return history
    
    def extract_conversation_topics(self, user_id: str) -> List[str]:
        """
        Extract main topics from a conversation.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            List[str]: Main conversation topics
        """
        if user_id not in self.conversations or not self.conversations[user_id]:
            return []
            
        # Combine all queries into a single text
        all_queries = " ".join([ex.get('user_query', '') for ex in self.conversations[user_id]])
        
        # Simple keyword extraction
        # In a real system, this would use NLP techniques
        common_words = [word.lower() for word in all_queries.split() 
                        if len(word) > 3 and word.lower() not in STOPWORDS]
        
        # Count word frequencies
        from collections import Counter
        word_counts = Counter(common_words)
        
        # Get top 5 topics
        return [word for word, _ in word_counts.most_common(5)]
        
    def get_relevant_documents(self, user_id: str) -> List[str]:
        """
        Get document IDs that were relevant in previous exchanges.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            List[str]: Document IDs that were useful in the conversation
        """
        if user_id not in self.conversations:
            return []
            
        # Collect all document IDs from context
        doc_ids = []
        for exchange in self.conversations[user_id]:
            context_docs = exchange.get('context_docs', [])
            for doc in context_docs:
                if 'id' in doc:
                    doc_ids.append(doc['id'])
                    
        # Return unique document IDs
        return list(set(doc_ids))


# Helper functions and constants

# Simple list of stopwords for topic extraction
STOPWORDS = {
    'the', 'and', 'is', 'in', 'it', 'to', 'that', 'of', 'for', 'with', 
    'this', 'are', 'on', 'not', 'be', 'as', 'what', 'how', 'why', 'when',
    'where', 'who', 'which', 'would', 'could', 'should', 'can', 'may', 'will',
    'have', 'has', 'had', 'been', 'was', 'were', 'there', 'their', 'they',
    'from', 'but', 'or', 'if', 'by', 'an', 'any', 'some', 'all', 'about'
}


class AdvancedFeaturesManager:
    """
    Main class for coordinating all advanced features.
    """
    
    def __init__(self, llm_service):
        """
        Initialize advanced features manager.
        
        Args:
            llm_service: Reference to LLM integration module
        """
        self.query_rewriter = QueryRewriter(llm_service)
        self.reliability_scorer = SourceReliabilityScorer()
        self.personalization = PersonalizationManager()
        self.conversation = ConversationManager()
        logger.info("AdvancedFeaturesManager initialized")
        
    def enhance_query(self, user_id: str, original_query: str) -> str:
        """
        Apply all query enhancement strategies.
        
        Args:
            user_id: Unique user identifier
            original_query: Original user query
            
        Returns:
            str: Enhanced query
        """
        # Get conversation context
        context = self.conversation.get_conversation_context(user_id, max_exchanges=3)
        
        # Step 1: Apply personalization
        personalized_query = self.personalization.personalize_query(user_id, original_query)
        
        # Step 2: Rewrite query with context
        enhanced_query = self.query_rewriter.rewrite_query(personalized_query, context)
        
        logger.info(f"Query enhancement: {original_query} -> {enhanced_query}")
        return enhanced_query
        
    def enhance_results(self, user_id: str, results: List[Dict]) -> List[Dict]:
        """
        Apply all result enhancement strategies.
        
        Args:
            user_id: Unique user identifier
            results: Original search results
            
        Returns:
            List[Dict]: Enhanced results
        """
        # Step 1: Score source reliability
        results = self.reliability_scorer.adjust_search_results(results)
        
        # Step 2: Apply personalization
        results = self.personalization.personalize_results(user_id, results)
        
        return results
        
    def store_interaction(self, user_id: str, query: str, response: str, 
                          context_docs: Optional[List[Dict]] = None) -> None:
        """
        Store interaction in conversation history.
        
        Args:
            user_id: Unique user identifier
            query: User query
            response: System response
            context_docs: Optional list of documents used for context
        """
        # Add to conversation history
        self.conversation.add_exchange(user_id, query, response, context_docs)
        
        # Update personalization profile
        # Extract topics from the query (simple implementation)
        topics = [word.lower() for word in query.split() 
                 if len(word) > 3 and word.lower() not in STOPWORDS]
        
        # Update profile with these topics
        topic_updates = {topic: 1 for topic in topics}
        self.personalization.update_profile(
            user_id, 
            {'history': {'topics': topic_updates}}
        )