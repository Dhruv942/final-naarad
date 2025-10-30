"""
Stage 3: LLM Article Filter and Ranker
Uses LLM reasoning to filter and rank articles based on alertsparse preferences.
Handles complex conditions like "only when team wins", semantic matching, etc.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import google.generativeai as genai

logger = logging.getLogger(__name__)


class LLMArticleRanker:
    """Filters and ranks articles using LLM reasoning."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        """Initialize with Gemini API."""
        self.api_key = api_key
        self.model_name = model
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        logger.info(f"Initialized LLM Article Ranker with model: {model}")
    
    async def filter_and_rank(
        self,
        alertsparse: Dict[str, Any],
        articles: List[Dict[str, Any]],
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Filter and rank articles based on alertsparse using LLM.
        
        Args:
            alertsparse: User preferences from Stage 1
            articles: Raw articles from Stage 2
            top_n: Number of top articles to return
            
        Returns:
            Dict with filtered_ranked_articles, included_count, excluded_count
        """
        try:
            if not articles:
                return {
                    "filtered_ranked_articles": [],
                    "included_count": 0,
                    "excluded_count": 0
                }
            
            logger.info(f"Filtering {len(articles)} articles for alert {alertsparse.get('alert_id')}")
            
            # Create LLM prompt
            prompt = self._create_ranking_prompt(alertsparse, articles, top_n)
            
            # Call LLM
            response = self.model.generate_content(prompt)
            result = self._extract_json_from_response(response.text)
            
            # Validate result
            filtered_articles = result.get("filtered_ranked_articles", [])
            
            final_result = {
                "filtered_ranked_articles": filtered_articles[:top_n],
                "included_count": len(filtered_articles),
                "excluded_count": len(articles) - len(filtered_articles)
            }
            
            logger.info(
                f"Filtered {len(articles)} → {final_result['included_count']} articles "
                f"(excluded: {final_result['excluded_count']})"
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in LLM ranking: {str(e)}")
            # Fallback: return most recent articles
            return self._fallback_ranking(articles, top_n)
    
    def _create_ranking_prompt(
        self,
        alertsparse: Dict[str, Any],
        articles: List[Dict[str, Any]],
        top_n: int
    ) -> str:
        """Create comprehensive LLM prompt for article filtering and ranking."""
        
        # Prepare article summaries (title + content snippet only)
        article_summaries = []
        for i, article in enumerate(articles):
            summary = {
                "index": i,
                "title": article.get("title", ""),
                "content_snippet": article.get("content", "")[:500],  # First 500 chars
                "source": article.get("source", ""),
                "published_date": str(article.get("published_date", ""))
            }
            article_summaries.append(summary)
        
        return f"""You are an AI assistant for a personalized news delivery system. Your job is to filter and rank news articles based on user preferences.

**USER PREFERENCES (from alertsparse):**
```json
{{
  "category": "{alertsparse.get('category', '')}",
  "sub_categories": {json.dumps(alertsparse.get('sub_categories', []))},
  "canonical_entities": {json.dumps(alertsparse.get('canonical_entities', []))},
  "event_conditions": {json.dumps(alertsparse.get('event_conditions', []))},
  "followup_questions": {json.dumps(alertsparse.get('followup_questions', []))},
  "custom_question": "{alertsparse.get('custom_question', '')}",
  "forbidden_topics": {json.dumps(alertsparse.get('forbidden_topics', []))}
}}
```

**CANDIDATE ARTICLES:**
```json
{json.dumps(article_summaries, indent=2, default=str)}
```

**YOUR TASK:**

1. **STRICTLY Filter Articles Based On:**
   - **Event Conditions (MOST CRITICAL)**: If event_conditions exist, validate them STRICTLY
     * Example: If event_conditions = [{{"type": "win", "entity": "india"}}], ONLY include articles where India actually WON
     * Example: If event_conditions = [{{"type": "loss", "entity": "rcb"}}], ONLY include articles where RCB LOST
     * Example: If event_conditions = [{{"type": "upcoming", "entity": "world cup"}}], ONLY include articles about FUTURE matches
     * DO NOT include articles that don't satisfy the event condition
   
   - **Canonical Entities**: Article MUST mention at least one canonical entity
   
   - **Forbidden Topics**: EXCLUDE any article containing forbidden topics
   
   - **Category/Sub-category**: Article must be relevant to the category

2. **Rank Remaining Articles By:**
   - Event condition satisfaction (100% weight if conditions exist)
   - Relevance to canonical entities (high weight)
   - Recency (newer is better)
   - Source credibility (Google News, major publishers preferred)
   - Match with followup questions

3. **Intelligence Requirements:**
   - Understand semantic meaning (e.g., "triumph", "victory" = win)
   - Handle negations and restrictions properly
   - Consider context (e.g., "India beat Australia" = India win)
   - Detect misleading titles

4. **Return Format:**
Return ONLY valid JSON (no markdown, no extra text):

```json
{{
  "filtered_ranked_articles": [
    {{
      "index": <original_index>,
      "title": "<article_title>",
      "url": "<article_url>",
      "source": "<source>",
      "published_date": "<date>",
      "relevance_score": <0.0-1.0>,
      "reason": "Brief explanation why this article was selected and ranked here"
    }}
  ],
  "excluded_count": <number>,
  "included_count": <number>
}}
```

**CRITICAL RULES:**
- If event_conditions exist, be VERY STRICT - only include articles that ACTUALLY satisfy the condition
- If no articles satisfy the criteria, return empty filtered_ranked_articles array
- Return at most {top_n} articles
- Sort by relevance_score (highest first)

Now analyze and filter the articles:"""
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        try:
            # Find JSON in response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx + 1]
                parsed = json.loads(json_str)
                
                # Validate structure
                if "filtered_ranked_articles" not in parsed:
                    raise ValueError("Missing filtered_ranked_articles")
                
                return parsed
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.error(f"Failed to extract JSON from LLM response: {str(e)}")
            logger.debug(f"Response was: {response_text[:500]}")
            return {
                "filtered_ranked_articles": [],
                "included_count": 0,
                "excluded_count": 0
            }
    
    def _fallback_ranking(
        self,
        articles: List[Dict[str, Any]],
        top_n: int
    ) -> Dict[str, Any]:
        """Simple fallback ranking when LLM fails."""
        try:
            # Sort by published date
            sorted_articles = sorted(
                articles,
                key=lambda x: x.get("published_date", datetime.min),
                reverse=True
            )
            
            # Take top N
            top_articles = sorted_articles[:top_n]
            
            # Format output
            formatted = []
            for i, article in enumerate(top_articles):
                formatted.append({
                    "index": i,
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "source": article.get("source", ""),
                    "published_date": str(article.get("published_date", "")),
                    "relevance_score": 0.5,
                    "reason": "Fallback ranking (LLM unavailable)"
                })
            
            return {
                "filtered_ranked_articles": formatted,
                "included_count": len(formatted),
                "excluded_count": len(articles) - len(formatted)
            }
            
        except Exception as e:
            logger.error(f"Error in fallback ranking: {str(e)}")
            return {
                "filtered_ranked_articles": [],
                "included_count": 0,
                "excluded_count": len(articles)
            }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test():
        ranker = LLMArticleRanker(api_key="YOUR_API_KEY")
        
        alertsparse = {
            "alert_id": "test123",
            "category": "sports",
            "sub_categories": ["cricket"],
            "canonical_entities": ["india", "indian cricket team", "virat kohli"],
            "event_conditions": [{"type": "win", "entity": "india"}],
            "followup_questions": ["world cup"],
            "custom_question": "only give me my team win",
            "forbidden_topics": ["football", "politics"]
        }
        
        articles = [
            {
                "title": "India wins World Cup final",
                "content": "India defeated Australia in a thrilling final...",
                "source": "ESPN",
                "url": "https://example.com/1",
                "published_date": datetime.utcnow()
            },
            {
                "title": "Australia beats India in practice match",
                "content": "Australia won the practice match...",
                "source": "Cricket.com",
                "url": "https://example.com/2",
                "published_date": datetime.utcnow()
            }
        ]
        
        result = await ranker.filter_and_rank(alertsparse, articles, top_n=3)
        
        print(json.dumps(result, indent=2, default=str))
    
    asyncio.run(test())
