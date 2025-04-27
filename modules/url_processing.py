# modules/url_processing.py
import os
import json
import re
import requests
import logging
import time
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from datetime import datetime
from bs4 import BeautifulSoup
from tqdm import tqdm

# ============= CONFIGURATION SETTINGS =============
URL_PROCESSING_CONFIG = {
    # URL Processing Settings
    'URLS_LIST': ["https://python.langchain.com/docs/how_to/"],
    'URLS_FILE': "urls.txt",
    'OUTPUT_DIR': "data/processed/urls",
    
    # Chunking Parameters
    'CHUNK_SIZE': 1000,
    'CHUNK_OVERLAP': 200,
    
    # Processing Options
    'USE_SELENIUM': False,
    'POLITENESS_DELAY': 1,
    'USER_AGENT': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    
    # Logging Configuration
    'LOG_LEVEL': logging.INFO,
    'LOG_FORMAT': '%(asctime)s - %(levelname)s: %(message)s',
    'LOG_DATE_FORMAT': '%Y-%m-%d %H:%M:%S'
}

# Configure logging
logging.basicConfig(
    level=URL_PROCESSING_CONFIG['LOG_LEVEL'], 
    format=URL_PROCESSING_CONFIG['LOG_FORMAT'],
    datefmt=URL_PROCESSING_CONFIG['LOG_DATE_FORMAT']
)

class URLProcessor:
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize URL processor with configuration
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = URL_PROCESSING_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # Ensure output directory exists
        os.makedirs(self.config['OUTPUT_DIR'], exist_ok=True)
    
    def validate_url(self, url: str) -> bool:
        """
        Validate if a URL is properly formatted
        
        Args:
            url: URL to validate
        
        Returns:
            Boolean indicating URL validity
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False
    
    def chunk_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.config['CHUNK_SIZE']
        chunk_overlap = chunk_overlap or self.config['CHUNK_OVERLAP']
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Find good breaking points
            breakpoints = [
                text.rfind('.', start + chunk_size // 2, end),
                text.rfind('?', start + chunk_size // 2, end),
                text.rfind('\n', start + chunk_size // 2, end)
            ]
            
            breakpoint = max(bp for bp in breakpoints if bp != -1)
            breakpoint = breakpoint if breakpoint != -1 else end
            
            chunks.append(text[start:breakpoint + 1])
            start = breakpoint + 1 - chunk_overlap
        
        return chunks
    
    def fetch_url_content(self, url: str) -> Optional[str]:
        """
        Fetch content from a URL
        
        Args:
            url: URL to fetch
        
        Returns:
            HTML content or None if fetch fails
        """
        if not self.validate_url(url):
            logging.error(f"Invalid URL format: {url}")
            return None
        
        try:
            headers = {'User-Agent': self.config['USER_AGENT']}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
            return None
    
    def extract_content(self, html_content: str, url: str) -> Dict[str, Any]:
        """
        Extract main content and metadata from HTML
        
        Args:
            html_content: HTML content
            url: Source URL
        
        Returns:
            Dictionary with content and metadata
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract metadata
            metadata = {
                'url': url,
                'domain': urlparse(url).netloc,
                'fetch_date': datetime.now().isoformat()
            }
            
            # Extract title
            if soup.title:
                metadata['title'] = soup.title.string.strip()
            
            # Remove unnecessary elements
            for unwanted in soup.select('script, style, nav, footer, header'):
                unwanted.extract()
            
            # Find main content
            main_content = soup.select_one('article, main, .content, #content') or soup.body
            content = main_content.get_text(separator='\n') if main_content else ''
            
            # Clean content
            content = re.sub(r'\n\s*\n', '\n\n', content)
            content = re.sub(r' +', ' ', content).strip()
            
            return {
                'content': content,
                'metadata': metadata
            }
        
        except Exception as e:
            logging.error(f"Content extraction error: {e}")
            return {'content': '', 'metadata': {}}
    
    def process_url(self, url: str) -> Optional[List[Dict[str, Any]]]:
        """
        Process a single URL
        
        Args:
            url: URL to process
        
        Returns:
            List of content chunks or None
        """
        logging.info(f"Processing URL: {url}")
        
        # Fetch content
        html_content = self.fetch_url_content(url)
        if not html_content:
            return None
        
        # Extract content
        extraction = self.extract_content(html_content, url)
        content = extraction['content']
        metadata = extraction['metadata']
        
        if not content:
            logging.warning(f"No content extracted from {url}")
            return None
        
        # Chunk content
        chunks = self.chunk_text(content)
        
        # Create chunk objects
        processed_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk = {
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "source": url,
                    "document_type": "web",
                    "chunk_number": i + 1,
                    "total_chunks": len(chunks)
                }
            }
            processed_chunks.append(chunk)
        
        logging.info(f"Extracted {len(processed_chunks)} chunks from {url}")
        
        # Politeness delay
        time.sleep(self.config.get('POLITENESS_DELAY', 1))
        
        return processed_chunks

def process_urls(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Process multiple URLs
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Processing results summary
    """
    processor = URLProcessor(config or {})
    
    # Determine URLs to process
    urls = processor.config.get('URLS_LIST', URL_PROCESSING_CONFIG['URLS_LIST'])
    output_dir = processor.config['OUTPUT_DIR']
    
    all_chunks = []
    processed_urls = []
    
    for url in tqdm(urls, desc="Processing URLs"):
        chunks = processor.process_url(url)
        
        if chunks:
            # Save chunks to individual file
            filename = f"{urlparse(url).netloc}_{datetime.now().strftime('%Y%m%d')}_chunks.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            
            processed_urls.append({
                'url': url,
                'chunks': len(chunks),
                'output_file': filepath
            })
            
            all_chunks.extend(chunks)
    
    return {
        'total_urls_processed': len(processed_urls),
        'total_chunks': len(all_chunks),
        'processed_urls': processed_urls
    }

# Standalone testing
if __name__ == "__main__":
    result = process_urls()
    print(f"Processed {result['total_urls_processed']} URLs")
    print(f"Total chunks: {result['total_chunks']}")