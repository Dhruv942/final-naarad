"""
Stage 1: LLM-Based Preference Parser
Converts user preferences into structured alertsparse format using LLM reasoning.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import google.generativeai as genai

logger = logging.getLogger(__name__)


class LLMPreferenceParser:
    """Parses user preferences using LLM to create intelligent alertsparse objects."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        """Initialize with Gemini API."""
        self.api_key = api_key
        self.model_name = model
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        logger.info(f"Initialized LLM Preference Parser with model: {model}")
    
    async def parse_user_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse user alert into structured alertsparse format using LLM.
        
        Args:
            alert_data: Raw user alert with category, sub_categories, followup_questions, custom_question
            
        Returns:
            Structured alertsparse object ready for DB storage
        """
        try:
            logger.info(f"Parsing alert: {alert_data.get('alert_id', 'unknown')}")
            
            # Extract basic fields
            category = alert_data.get("main_category", alert_data.get("category", "")).lower()
            sub_categories = alert_data.get("sub_categories", [])
            followup_questions = alert_data.get("followup_questions", [])
            custom_question = alert_data.get("custom_question", "")
            
            # Create LLM prompt
            prompt = self._create_parsing_prompt(
                category, sub_categories, followup_questions, custom_question
            )
            
            # Call LLM
            response = self.model.generate_content(prompt)
            parsed_data = self._extract_json_from_response(response.text)
            
            # Build final alertsparse object
            alertsparse = {
                "alert_id": alert_data.get("alert_id"),
                "user_id": alert_data.get("user_id"),
                "category": category,
                "sub_categories": [s.lower() for s in sub_categories],
                "followup_questions": followup_questions,
                "custom_question": custom_question,
                
                # LLM-generated fields
                "canonical_entities": parsed_data.get("canonical_entities", []),
                "event_conditions": parsed_data.get("event_conditions", []),
                "contextual_query": parsed_data.get("contextual_query", ""),
                "forbidden_topics": parsed_data.get("forbidden_topics", []),
                
                # Metadata
                "created_at": datetime.utcnow(),
                "last_updated": datetime.utcnow(),
                "parsing_version": "llm_v1"
            }
            
            logger.info(f"Successfully parsed alert {alertsparse['alert_id']}")
            logger.debug(f"Canonical entities: {alertsparse['canonical_entities']}")
            logger.debug(f"Event conditions: {alertsparse['event_conditions']}")
            
            return alertsparse
            
        except Exception as e:
            logger.error(f"Error parsing alert with LLM: {str(e)}")
            # Return basic alertsparse on failure
            return self._create_fallback_alertsparse(alert_data)
    
    def _create_parsing_prompt(
        self, 
        category: str, 
        sub_categories: List[str], 
        followup_questions: List[str],
        custom_question: str
    ) -> str:
        """Create the LLM prompt for preference parsing."""
        return f"""You are an AI assistant for a personalized news delivery system. Your job is to parse user preferences into structured data.

**User Preferences:**
- **Category**: {category}
- **Sub-categories**: {', '.join(sub_categories) if sub_categories else 'None'}
- **Follow-up Questions**: {', '.join(followup_questions) if followup_questions else 'None'}
- **Custom Question**: {custom_question if custom_question else 'None'}

**Your Task:**
Extract and infer the following information:

1. **canonical_entities**: Main entities (people, teams, organizations, locations, events) the user cares about
   - Examples: "Team India", "Virat Kohli", "World Cup", "RCB", "IPL"
   - Include synonyms and common variations

2. **event_conditions**: Specific event types or conditions the user wants to track
   - Examples: {{"type": "win", "entity": "india"}}, {{"type": "loss", "entity": "rcb"}}, {{"type": "upcoming", "entity": "world cup"}}
   - Types: win, loss, upcoming, score, injury, transfer, record, general
   - Pay special attention to restrictive phrases like "only when win", "just losses", etc.

3. **contextual_query**: An expanded search query with relevant keywords and synonyms
   - Include the category, entities, related terms, and synonyms
   - Make it comprehensive for news search (Google News, RSS feeds)
   - Example: "india cricket team win victory match result virat kohli world cup test odi t20"

4. **forbidden_topics**: Topics the user wants to EXCLUDE
   - Based on category (e.g., if user wants cricket, exclude football, basketball, etc.)
   - Based on negative phrases in custom question

**CRITICAL INTELLIGENCE:**
- If custom_question contains "only" or "just" or "when" + event → Set strict event_conditions
- Example: "only give me my team win" → event_conditions: [{{"type": "win", "entity": "india"}}]
- Example: "upcoming matches only" → event_conditions: [{{"type": "upcoming", "entity": "<team>"}}]

**Output Format:**
Return ONLY valid JSON (no markdown, no explanation):

{{
  "canonical_entities": ["entity1", "entity2", ...],
  "event_conditions": [{{"type": "event_type", "entity": "entity_name"}}, ...],
  "contextual_query": "expanded search query with keywords",
  "forbidden_topics": ["topic1", "topic2", ...]
}}

**Examples:**

Input: category="sports", sub_categories=["cricket"], followup_questions=["team india"], custom_question="only give me my team win"
Output:
{{
  "canonical_entities": ["india", "indian cricket team", "team india", "men in blue", "bcci"],
  "event_conditions": [{{"type": "win", "entity": "india"}}],
  "contextual_query": "india cricket team win victory match result triumph defeat opponent score international test odi t20 world cup",
  "forbidden_topics": ["football", "basketball", "tennis", "hockey", "politics", "entertainment"]
}}

Input: category="sports", sub_categories=["football"], followup_questions=["premier league", "manchester united"], custom_question="upcoming fixtures"
Output:
{{
  "canonical_entities": ["manchester united", "man utd", "red devils", "premier league", "epl"],
  "event_conditions": [{{"type": "upcoming", "entity": "manchester united"}}],
  "contextual_query": "manchester united man utd premier league upcoming fixture schedule next match epl football soccer",
  "forbidden_topics": ["cricket", "basketball", "tennis", "politics"]
}}

Now parse the user preferences provided above:"""
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        try:
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx + 1]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.error(f"Failed to extract JSON from LLM response: {str(e)}")
            logger.debug(f"Response was: {response_text[:500]}")
            return {
                "canonical_entities": [],
                "event_conditions": [],
                "contextual_query": "",
                "forbidden_topics": []
            }
    
    def _create_fallback_alertsparse(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic alertsparse when LLM fails."""
        category = alert_data.get("main_category", alert_data.get("category", "")).lower()
        sub_categories = alert_data.get("sub_categories", [])
        
        return {
            "alert_id": alert_data.get("alert_id"),
            "user_id": alert_data.get("user_id"),
            "category": category,
            "sub_categories": [s.lower() for s in sub_categories],
            "followup_questions": alert_data.get("followup_questions", []),
            "custom_question": alert_data.get("custom_question", ""),
            "canonical_entities": sub_categories,
            "event_conditions": [],
            "contextual_query": f"{category} {' '.join(sub_categories)}",
            "forbidden_topics": [],
            "created_at": datetime.utcnow(),
            "last_updated": datetime.utcnow(),
            "parsing_version": "fallback_v1"
        }
