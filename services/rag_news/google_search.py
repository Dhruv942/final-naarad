"""
Google News Search Module

This module provides functionality to search for news articles using Google News.
It handles API calls, rate limiting, and result formatting.
"""

import os
import json
import time
import asyncio
import logging
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from urllib.parse import urlencode, quote_plus
from .config import GOOGLE_API_KEY, GOOGLE_CX

# Configure logging
logger = logging.getLogger(__name__)

class GoogleNewsSearch:
    """Handles Google News search operations"""
    
    def __init__(self, api_key: str = None, cx: str = None):
        """
        Initialize the GoogleNewsSearch instance.
        
        Args:
            api_key: Google Custom Search JSON API key (optional, will use config if not provided)
            cx: Custom Search Engine ID (optional, will use config if not provided)
        """
        self.api_key = api_key or GOOGLE_API_KEY
        self.cx = cx or GOOGLE_CX
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.rate_limit_delay = 1  # seconds between requests
        self.last_request_time = 0
        
        if not self.api_key or not self.cx:
            logger.error("API key or CX not configured for Google News search")
            raise ValueError("Google API key and CX must be configured in config.py")
    
    async def search(
        self,
        query: str,
        num_results: int = 10,
        days_back: int = 1,
        language: str = "en",
        region: str = "us"
    ) -> List[Dict[str, Any]]:
        """
        Search for news articles matching the query.
        
        Args:
            query: Search query string
            num_results: Maximum number of results to return (max 10 per request)
            days_back: Only return results from the last N days
            language: Language code (e.g., 'en', 'es')
            region: Region code (e.g., 'us', 'uk')
            
        Returns:
            List of article dictionaries with metadata
        """
        if not self.api_key or not self.cx:
            logger.error("API key or CX not configured for Google News search")
            return []
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for Google's API
        date_range = f"{start_date.strftime('%Y%m%d')}:{end_date.strftime('%Y%m%d')}"
        
        # Prepare query parameters
        q = (query or "").strip()
        site = None
        # Extract first site:domain.com from the query to use siteSearch param
        import re
        m = re.search(r"site:([\w.-]+)", q)
        if m:
            site = m.group(1)
            # Remove site:domain from query string to avoid duplication
            q = re.sub(r"site:[\w.-]+", "", q).strip()

        params = {
            'q': q,
            'key': self.api_key,
            'cx': self.cx,
            'num': min(num_results, 10),
            'dateRestrict': f'd{max(1, days_back)}',
            'safe': 'off',
        }
        if site:
            params['siteSearch'] = site
            params['siteSearchFilter'] = 'i'
        
        try:
            # Respect rate limiting
            await self._enforce_rate_limit()
            
            # Make the request
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            # Parse and format results
            data = response.json()
            results = data.get('items', [])
            return [self._format_result(item) for item in results[:num_results]]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching Google News: {e}")
            return []
    
    def _format_result(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Format a Google News API result into a standardized format"""
        link = item.get('link', '')
        display = item.get('displayLink') or ''
        return {
            'title': item.get('title', ''),
            'snippet': item.get('snippet', ''),
            'url': link,
            'source': display,
            'published_at': datetime.utcnow().isoformat(),
            'image_url': item.get('pagemap', {}).get('cse_image', [{}])[0].get('src', '')
        }
    
    @staticmethod
    def _parse_date(date_str: str) -> str:
        """Parse date string from Google News result"""
        try:
            # Try to extract date from the snippet
            # This is a simplified example - you might need to adjust based on actual format
            return datetime.utcnow().isoformat()
        except (ValueError, AttributeError):
            return datetime.utcnow().isoformat()
    
    async def _enforce_rate_limit(self):
        """Ensure we don't exceed API rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()


# Singleton instance
_google_news_search = None

def get_google_news_search(api_key: str = None, cx: str = None) -> GoogleNewsSearch:
    """Get or create a GoogleNewsSearch instance"""
    global _google_news_search
    if _google_news_search is None:
        _google_news_search = GoogleNewsSearch(api_key, cx)
    return _google_news_search


# For backward compatibility
async def search_google_news(
    query: str,
    num_results: int = 10,
    days_back: int = 1,
    language: str = "en",
    region: str = "us"
) -> List[Dict[str, Any]]:
    """
    Search Google News (legacy interface)
    
    Args:
        query: Search query
        num_results: Max number of results to return
        days_back: Only return results from the last N days
        language: Language code (e.g., 'en', 'es')
        region: Region code (e.g., 'us', 'uk')
        
    Returns:
        List of news article dictionaries
    """
    searcher = get_google_news_search()
    return await searcher.search(query, num_results, days_back, language, region)