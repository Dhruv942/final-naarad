"""
LLM-based article ranking and filtering for RAG news system.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

# Assuming you have a config with your LLM settings
from .config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

class LLMArticleRanker:
    """Handles LLM-based article ranking and filtering."""
    
    def __init__(self, llm_client=None):
        """Initialize with an optional LLM client (for testing)."""
        self.llm_client = llm_client
        
    async def rank_and_filter_articles(
        self,
        alertsparse_data: Dict[str, Any],
        articles: List[Dict[str, Any]],
        max_articles: int = 20
    ) -> Dict[str, Any]:
        """
        Rank and filter articles based on user preferences using LLM.
        
        Args:
            alertsparse_data: User preferences from alertsparse collection
            articles: List of candidate articles from RAG retrieval
            max_articles: Maximum number of articles to return
            
        Returns:
            Dict with filtered and ranked articles
        """
        if not articles:
            return {
                "filtered_ranked_articles": [],
                "excluded_count": 0,
                "included_count": 0
            }
        
        try:
            # Prepare the prompt
            prompt = self._create_ranking_prompt(alertsparse_data, articles)
            
            # Call LLM (using Gemini API as per config)
            response = await self._call_llm(prompt)
            
            # Parse and validate the response
            result = self._parse_llm_response(response)
            
            # Ensure we don't exceed max_articles
            if len(result["filtered_ranked_articles"]) > max_articles:
                result["filtered_ranked_articles"] = result["filtered_ranked_articles"][:max_articles]
                
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM ranking: {str(e)}")
            # Fallback: return original articles with default ranking
            return self._fallback_ranking(articles, max_articles)
    
    def _create_ranking_prompt(
        self,
        alertsparse_data: Dict[str, Any],
        articles: List[Dict[str, Any]]
    ) -> str:
        """Create the LLM prompt for article ranking."""
        # Prepare the user preferences section
        user_prefs = {
            "category": alertsparse_data.get("category", ""),
            "sub_categories": alertsparse_data.get("sub_categories", []),
            "canonical_entities": alertsparse_data.get("canonical_entities", []),
            "followup_questions": alertsparse_data.get("followup_questions", []),
            "custom_question": alertsparse_data.get("custom_question", ""),
            "event_conditions": alertsparse_data.get("event_conditions", [])
        }
        
        # Prepare the prompt
        prompt = f"""You are an AI assistant for a personalized news delivery system.

User preferences object:
{user_prefs_json}

Candidate news articles from RAG search:
{articles_json}

Your job:
1. Understand the user's preferences:
   - category: {user_prefs['category']}
   - sub_categories: {', '.join(user_prefs['sub_categories'])}
   - followup_questions: {', '.join(user_prefs['followup_questions'])}
   - custom_question: {user_prefs['custom_question']}
   - event_conditions: {user_prefs['event_conditions']}

2. Apply intelligent filtering:
   - Only include articles relevant to canonical entities: {', '.join(user_prefs['canonical_entities'])}
   - If event_conditions exist, validate them strictly
   - Respect negative or restrictive user intent

3. Rank relevant articles by:
   - Event condition satisfaction (MOST IMPORTANT)
   - Followup relevance
   - Positive relevance to user interest
   - Recency (newer is better)
   - Source credibility (Google-first > top publishers > others)

4. Remove:
   - Duplicate URLs
   - Unrelated sports/categories
   - Articles without contextual match

5. Output STRICT JSON with filtered articles sorted by relevance_score (0-1):
{{
  "filtered_ranked_articles": [
    {{
      "title": "...",
      "url": "...",
      "published_date": "...",
      "reason": "why this was selected",
      "relevance_score": 0.95
    }}
  ],
  "excluded_count": 0,
  "included_count": 0
}}"""
        
        return prompt.format(
            user_prefs_json=json.dumps(user_prefs, indent=2),
            articles_json=json.dumps(articles, indent=2, default=str)
        )
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        # If we have a mock client for testing, use it
        if self.llm_client:
            return await self.llm_client.generate(prompt)
            
        # Otherwise, use Gemini API
        import google.generativeai as genai
        
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        
        try:
            response = await asyncio.to_thread(
                model.generate_content,
                prompt
            )
            return response.text
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            raise
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM response into the expected format."""
        try:
            # Try to extract JSON from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                
                # Validate the structure
                if not all(k in result for k in ["filtered_ranked_articles", "excluded_count", "included_count"]):
                    raise ValueError("Invalid response format from LLM")
                    
                return result
            
            raise ValueError("No valid JSON found in response")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            raise ValueError("Invalid JSON response from LLM") from e
    
    def _fallback_ranking(
        self,
        articles: List[Dict[str, Any]],
        max_articles: int
    ) -> Dict[str, Any]:
        """Fallback ranking when LLM fails."""
        # Simple recency-based fallback
        sorted_articles = sorted(
            articles,
            key=lambda x: x.get("published_date", ""),
            reverse=True
        )
        
        return {
            "filtered_ranked_articles": sorted_articles[:max_articles],
            "excluded_count": max(0, len(articles) - max_articles),
            "included_count": min(len(articles), max_articles)
        }

# Example usage:
async def example_usage():
    # Example alertsparse data
    alertsparse_data = {
        "category": "sports",
        "sub_categories": ["cricket", "team_india"],
        "canonical_entities": ["india", "virat kohli", "world cup"],
        "followup_questions": ["world cup matches", "virat kohli performance"],
        "custom_question": "only show me when India wins",
        "event_conditions": [{"type": "win", "entity": "india"}]
    }
    
    # Example articles (would come from your RAG system)
    example_articles = [
        {
            "title": "India wins the World Cup 2023",
            "url": "https://example.com/india-wins",
            "published_date": "2023-11-19T14:30:00Z",
            "content": "India has won the Cricket World Cup 2023 after a thrilling final..."
        },
        {
            "title": "Upcoming match: India vs Australia",
            "url": "https://example.com/ind-vs-aus",
            "published_date": "2023-11-20T09:15:00Z",
            "content": "India will face Australia in the next match..."
        }
    ]
    
    # Create ranker and process articles
    ranker = LLMArticleRanker()
    result = await ranker.rank_and_filter_articles(alertsparse_data, example_articles)
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
