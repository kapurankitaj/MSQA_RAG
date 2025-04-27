import os
import re
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import logging

# ============= CONFIGURATION SETTINGS =============
CONFIG = {
    # Citation format settings
    "citation": {
        "style": "inline",  # Options: "inline", "footnote", "end_references"
        "include_page_numbers": True,  # Include page numbers when available
        "default_format": "{index}. {title}, {source_type}",
        "max_length": 100,  # Maximum length for citation text
        "marker_format": "[{index}]"  # Format for inline citation markers
    },
    
    # Citation database settings
    "database": {
        "path": "data/citations",
        "log_file": "citation_log.json",  # Log of all generated citations
        "verification_threshold": 0.3  # Threshold for source verification confidence
    },
    
    # Response formats
    "response_formats": {
        "footnote": """
---
Sources:
{footnotes}
""",
        "end_references": """
---
References:
{references}
"""
    },
    
    # Logging settings
    "logging": {
        "level": logging.INFO,
        "format": '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        "log_file": "citation_system.log"
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
logger = logging.getLogger("CitationSystem")

class CitationManager:
    """
    Manages citations for retrieved documents in the RAG system.
    Handles citation generation, tracking, and verification.
    """
    
    def __init__(self, config=CONFIG):
        """
        Initialize the Citation Manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.citation_style = config["citation"]["style"]
        self.citation_db_path = config["database"]["path"]
        
        # Create citation database directory if it doesn't exist
        os.makedirs(self.citation_db_path, exist_ok=True)
        
        # Load citation log if it exists
        self.citation_log_path = os.path.join(self.citation_db_path, config["database"]["log_file"])
        self.citation_log = self._load_citation_log()
        
        logger.info(f"Initialized Citation Manager with style: {self.citation_style}")
    
    def _load_citation_log(self) -> Dict[str, Any]:
        """Load the citation log from disk or create a new one."""
        if os.path.exists(self.citation_log_path):
            try:
                with open(self.citation_log_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading citation log: {e}")
                return {"citations": [], "meta": {"created": datetime.now().isoformat()}}
        else:
            return {"citations": [], "meta": {"created": datetime.now().isoformat()}}
    
    def _save_citation_log(self):
        """Save the citation log to disk."""
        try:
            with open(self.citation_log_path, 'w', encoding='utf-8') as f:
                json.dump(self.citation_log, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving citation log: {e}")
    
    def generate_source_id(self, source_info: Dict[str, Any]) -> str:
        """
        Generate a unique ID for a source based on its metadata
        
        Args:
            source_info: Dictionary containing source metadata
            
        Returns:
            Unique source ID
        """
        # Create a string representation of key source information
        source_string = json.dumps({
            k: v for k, v in sorted(source_info.items())
            if k in ['source', 'title', 'source_file', 'url']
        })
        
        # Generate a hash of the source string
        return hashlib.md5(source_string.encode()).hexdigest()[:12]
    
    def format_citation(self, source_info: Dict[str, Any], citation_index: int) -> Dict[str, Any]:
        """
        Format a citation based on source information
        
        Args:
            source_info: Dictionary containing source metadata
            citation_index: Index number for the citation
            
        Returns:
            Dictionary with formatted citation information
        """
        # Extract metadata fields with fallbacks
        title = source_info.get('title', 'Untitled Document')
        source = source_info.get('source', source_info.get('source_file', 'Unknown Source'))
        source_type = source_info.get('document_type', 'document')
        page = source_info.get('page_number', None)
        page_info = f", p. {page}" if page and self.config["citation"]["include_page_numbers"] else ""
        chunk_number = source_info.get('chunk_number', None)
        date = source_info.get('date', source_info.get('extraction_date', ''))
        
        # Generate citation text
        try:
            citation_text = self.config["citation"]["default_format"].format(
                index=citation_index,
                title=title,
                source=source,
                source_type=source_type,
                page_info=page_info,
                date=date
            )
        except Exception as e:
            logger.warning(f"Error formatting citation: {e}")
            citation_text = f"{citation_index}. {source}"
        
        # Truncate if too long
        max_length = self.config["citation"]["max_length"]
        if len(citation_text) > max_length:
            citation_text = citation_text[:max_length-3] + "..."
        
        # Generate citation marker
        citation_marker = self.config["citation"]["marker_format"].format(index=citation_index)
        
        # Generate source ID
        source_id = self.generate_source_id(source_info)
        
        return {
            "id": source_id,
            "index": citation_index,
            "marker": citation_marker,
            "text": citation_text,
            "source_info": source_info,
            "timestamp": datetime.now().isoformat()
        }
    
    def process_sources(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a list of sources to generate citation information
        
        Args:
            sources: List of dictionaries containing source metadata
            
        Returns:
            Dictionary with citation information
        """
        if not sources:
            return {"citations": [], "style": self.citation_style}
        
        # Track unique sources using source_id
        unique_sources = {}
        
        # Process each source
        citations = []
        for i, source in enumerate(sources, 1):
            # Get metadata from the source
            metadata = source.get('metadata', {})
            if not metadata:
                logger.warning(f"Source {i} has no metadata")
                continue
            
            # Store source text for verification
            if 'text' in source and 'text' not in metadata:
                metadata['text'] = source['text']
            
            # Format the citation
            citation = self.format_citation(metadata, i)
            
            # Track unique sources
            source_id = citation['id']
            if source_id not in unique_sources:
                unique_sources[source_id] = citation
                citations.append(citation)
            
            # Log the citation
            self._log_citation(citation)
        
        # Return different formats based on citation style
        return {
            "citations": citations,
            "style": self.citation_style,
            "unique_count": len(unique_sources)
        }
    
    def _log_citation(self, citation: Dict[str, Any]):
        """Add a citation to the log."""
        self.citation_log["citations"].append(citation)
        self.citation_log["meta"]["last_updated"] = datetime.now().isoformat()
        self.citation_log["meta"]["count"] = len(self.citation_log["citations"])
        self._save_citation_log()
    
    def format_response_with_citations(
        self, 
        response_text: str, 
        citations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format a response with citations according to the citation style
        
        Args:
            response_text: Original response text
            citations: Citation information from process_sources
            
        Returns:
            Dictionary with formatted response and citation information
        """
        citation_list = citations.get("citations", [])
        style = citations.get("style", self.citation_style)
        
        if not citation_list:
            return {"text": response_text, "citations": []}
        
        # Format response based on citation style
        if style == "inline":
            formatted_response = self._format_inline_citations(response_text, citation_list)
        elif style == "footnote":
            formatted_response = self._format_footnote_citations(response_text, citation_list)
        elif style == "end_references":
            formatted_response = self._format_end_references(response_text, citation_list)
        else:
            # Default to inline if unknown style
            formatted_response = self._format_inline_citations(response_text, citation_list)
        
        return {
            "text": formatted_response,
            "citations": citation_list,
            "style": style
        }
    
    def _format_inline_citations(
        self, 
        response_text: str, 
        citations: List[Dict[str, Any]]
    ) -> str:
        """Format response with inline citations."""
        # Split text into paragraphs
        paragraphs = response_text.split('\n\n')
        
        # Distribute citations among paragraphs more intelligently
        # First, find paragraphs that match content from each source
        citation_assignments = {}
        
        for citation in citations:
            source_text = citation["source_info"].get("text", "").lower()
            source_id = citation["id"]
            marker = citation["marker"]
            
            # Try to find the most relevant paragraph for this citation
            best_match_index = -1
            best_match_score = 0
            
            for i, paragraph in enumerate(paragraphs):
                if not paragraph.strip():
                    continue
                    
                # Skip paragraphs that already have citations
                if any(i == assigned_para for assigned_para, _ in citation_assignments.values()):
                    continue
                
                # Simple matching score based on overlapping words
                para_words = set(self._normalize_text(paragraph).split())
                source_words = set(self._normalize_text(source_text).split())
                
                if para_words and source_words:
                    overlap = len(para_words.intersection(source_words))
                    score = overlap / len(para_words)
                    
                    if score > best_match_score:
                        best_match_score = score
                        best_match_index = i
            
            # If we found a good match, assign the citation to that paragraph
            if best_match_index >= 0:
                citation_assignments[source_id] = (best_match_index, marker)
            # Otherwise, assign to the first paragraph without a citation
            else:
                for i, paragraph in enumerate(paragraphs):
                    if paragraph.strip() and not any(i == assigned_para for assigned_para, _ in citation_assignments.values()):
                        citation_assignments[source_id] = (i, marker)
                        break
        
        # Apply citation markers to paragraphs
        for source_id, (para_index, marker) in citation_assignments.items():
            if 0 <= para_index < len(paragraphs):
                paragraphs[para_index] = paragraphs[para_index].rstrip() + " " + marker
        
        # Add reference list at the end
        formatted_text = '\n\n'.join(paragraphs)
        references = '\n'.join([citation["text"] for citation in citations])
        formatted_text += f"\n\n---\nSources:\n{references}"
        
        return formatted_text
    
    def _format_footnote_citations(
        self, 
        response_text: str, 
        citations: List[Dict[str, Any]]
    ) -> str:
        """Format response with footnote citations."""
        # First use the improved inline citation formatting logic
        formatted_text = self._format_inline_citations(response_text, citations)
        
        # Then replace the reference section with footnote format
        parts = formatted_text.split("---\nSources:")
        if len(parts) == 2:
            footnotes = parts[1].strip()
            formatted_text = parts[0] + self.config["response_formats"]["footnote"].format(footnotes=footnotes)
        
        return formatted_text
    
    def _format_end_references(
        self, 
        response_text: str, 
        citations: List[Dict[str, Any]]
    ) -> str:
        """Format response with end references."""
        # Create references list in the expected format
        references = '\n'.join([
            f"{citation['index']}. {citation['source_info'].get('title', 'Untitled')}, "
            f"{citation['source_info'].get('source', 'Unknown Source')}"
            for citation in citations
        ])
        
        # Add the references to the response
        return response_text + self.config["response_formats"]["end_references"].format(references=references)
    
    def verify_citations(
        self, 
        response_text: str, 
        citations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify that citations are properly used in the response
        
        Args:
            response_text: Response text to verify
            citations: List of citation information
            
        Returns:
            Dictionary with verification results
        """
        verification_results = []
    
        for citation in citations:
            marker = citation["marker"]
            source_id = citation["id"]
            
            # Check if citation marker appears in text
            # Use strict pattern matching to find the exact marker
            marker_pattern = re.escape(marker)
            marker_matches = re.findall(marker_pattern, response_text)
            marker_present = len(marker_matches) > 0
            
            # Log for debugging
            logger.debug(f"Checking marker '{marker}' in text: {'FOUND' if marker_present else 'NOT FOUND'}")
            
            # Get source text for content verification
            source_info = citation["source_info"]
            source_text = source_info.get("text", "")
            
            # Extract key phrases from source text using improved extraction
            key_phrases = self._extract_key_phrases(source_text)
            
            # Check for presence of key phrases in response with more flexible matching
            phrase_matches = []
            for phrase in key_phrases:
                # Make matching more lenient by normalizing text
                normalized_phrase = self._normalize_text(phrase)
                normalized_response = self._normalize_text(response_text)
                
                # Check for substantial overlap rather than exact matches
                if len(normalized_phrase) > 5 and normalized_phrase in normalized_response:
                    phrase_matches.append(phrase)
            
            # Ensure at least minimal matching for short phrases
            if not phrase_matches and key_phrases and len(key_phrases[0]) > 3:
                # Try matching just the first few words of the first phrase
                first_phrase_start = ' '.join(key_phrases[0].split()[:3])
                if first_phrase_start.lower() in response_text.lower():
                    phrase_matches.append(first_phrase_start)
            
            # Calculate verification confidence with better formula
            # If marker is present, that's a positive signal
            if marker_present:
                base_confidence = 0.5
            else:
                base_confidence = 0
                
            # Add additional confidence based on content matches
            match_ratio = len(phrase_matches) / max(len(key_phrases), 1) if key_phrases else 0
            content_confidence = match_ratio * 0.5  # Content matching contributes up to 50% of confidence
            
            # Combine for final confidence score
            confidence = base_confidence + content_confidence
            
            verification_results.append({
                "citation_id": source_id,
                "marker_present": marker_present,
                "content_verified": confidence >= self.config["database"]["verification_threshold"],
                "confidence": confidence,
                "matched_phrases": len(phrase_matches),
                "total_phrases": len(key_phrases)
            })
        
        return {
            "verified": all(result["content_verified"] for result in verification_results),
            "results": verification_results
        }
      
    def _normalize_text(self, text: str) -> str:
        """Normalize text for more flexible matching."""
        if not text:
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove punctuation that might interfere with matching
        text = re.sub(r'[.,;:!?()"\'-]', ' ', text)
        # Normalize whitespace again after punctuation removal
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_key_phrases(self, text: str, max_phrases: int = 5) -> List[str]:
        """
        Extract key phrases from text for verification - improved version
        
        Args:
            text: Source text
            max_phrases: Maximum number of phrases to extract
            
        Returns:
            List of key phrases
        """
        if not text:
            return []
        
        # First try: extract sentences and noun phrases
        sentences = []
        for sentence in re.split(r'(?<=[.!?])\s+', text):
            if len(sentence.strip()) > 10:
                sentences.append(sentence.strip())
        
        # If we don't have enough sentences, break by commas too
        if len(sentences) < 2:
            comma_phrases = [p.strip() for p in text.split(',') if len(p.strip()) > 10]
            sentences.extend(comma_phrases)
        
        # If still not enough, just break by length
        if len(sentences) < 2 and len(text) > 20:
            chunks = []
            words = text.split()
            for i in range(0, len(words), 5):
                chunk = ' '.join(words[i:i+5])
                if chunk:
                    chunks.append(chunk)
            sentences.extend(chunks)
        
        # Deduplicate and limit
        unique_phrases = []
        for phrase in sentences:
            if phrase not in unique_phrases:
                unique_phrases.append(phrase)
                if len(unique_phrases) >= max_phrases:
                    break
        
        # Create shorter variants too for more flexible matching
        final_phrases = list(unique_phrases)
        for phrase in unique_phrases:
            words = phrase.split()
            if len(words) > 3:
                # Add a version with just the first 3-4 words
                short_phrase = ' '.join(words[:min(4, len(words))])
                if short_phrase not in final_phrases:
                    final_phrases.append(short_phrase)
        
        return final_phrases[:max_phrases]

def create_sources_from_retrieval_results(retrieval_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert retrieval results into source format for citation manager
    
    Args:
        retrieval_results: Results from vector or keyword search
        
    Returns:
        List of sources in the format expected by citation manager
    """
    sources = []
    
    for result in retrieval_results:
        # Extract metadata
        metadata = result.get('metadata', {})
        
        # Create source dict
        source = {
            'text': result.get('text', ''),
            'metadata': metadata
        }
        
        sources.append(source)
    
    return sources

def test_citation_system():
    """Test the citation system with sample data."""
    # Create sample sources with more detailed text that better overlaps with the response
    sample_sources = [
        {
            'text': 'Retrieval Augmented Generation (RAG) is an AI technique that enhances language models by incorporating external knowledge during generation.',
            'metadata': {
                'title': 'Introduction to RAG',
                'source': 'AI Documentation',
                'document_type': 'technical_report',
                'page_number': 12,
                'chunk_number': 3,
                'total_chunks': 8
            }
        },
        {
            'text': 'Vector databases store embeddings for efficient similarity search using algorithms like FAISS. They play a crucial role in RAG systems by enabling fast retrieval of relevant documents.',
            'metadata': {
                'title': 'Vector Databases Explained',
                'source': 'Database Journal',
                'document_type': 'article',
                'url': 'https://example.com/vector-db'
            }
        }
    ]
    
    # Create sample response that clearly uses content from the sources
    sample_response = """
Retrieval Augmented Generation (RAG) is an AI technique that enhances language models by incorporating external knowledge.

Vector databases play a crucial role in RAG systems by enabling efficient similarity search for relevant context documents.
"""
    
    # Test inline citation style
    print("Testing inline citation style...")
    citation_manager = CitationManager()
    citations = citation_manager.process_sources(sample_sources)
    formatted_response = citation_manager.format_response_with_citations(sample_response, citations)
    print("\nFormatted response with inline citations:")
    print(formatted_response['text'])
    
    # Test verification on the INLINE format which contains citation markers
    print("\nTesting citation verification on inline format...")
    inline_verification = citation_manager.verify_citations(formatted_response['text'], citations['citations'])
    print(f"Verification results: {json.dumps(inline_verification, indent=2)}")
    
    # Test footnote citation style
    print("\nTesting footnote citation style...")
    config_footnote = CONFIG.copy()
    config_footnote["citation"]["style"] = "footnote"
    citation_manager = CitationManager(config=config_footnote)
    citations = citation_manager.process_sources(sample_sources)
    formatted_response = citation_manager.format_response_with_citations(sample_response, citations)
    print("\nFormatted response with footnote citations:")
    print(formatted_response['text'])
    
    # Test end references citation style
    print("\nTesting end references citation style...")
    config_end_ref = CONFIG.copy()
    config_end_ref["citation"]["style"] = "end_references"
    citation_manager = CitationManager(config=config_end_ref)
    citations = citation_manager.process_sources(sample_sources)
    formatted_response = citation_manager.format_response_with_citations(sample_response, citations)
    print("\nFormatted response with end references:")
    print(formatted_response['text'])
    
    return {
        "citations": citations,
        "formatted_response": formatted_response,
        "verification": inline_verification  # Return the inline verification results
    }

# Run the test when executed as a script
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("CITATION SYSTEM DEMONSTRATION")
    print("=" * 50)
    test_citation_system()