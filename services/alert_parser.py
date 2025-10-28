"""
Alert Parser for converting raw user alerts into structured alert objects.
Handles entity extraction, event condition parsing, and query generation.
"""

import re
import json
from typing import Dict, List, Any, Optional, Set
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Event types and their keyword mappings
event_keywords = {
    "win": ["win", "victory", "beat", "triumph", "defeat"],
    "loss": ["loss", "defeat", "lose", "beaten", "fall", "drop"],
    "upcoming": ["upcoming", "schedule", "fixture", "next match", "when"],
    "score": ["score", "result", "update", "live", "latest"],
    "record": ["record", "milestone", "achievement", "century", "half-century"],
    "injury": ["injury", "injured", "hurt", "out of", "ruled out"],
    "transfer": ["transfer", "signing", "signed", "deal", "contract"],
}

# Common entity synonyms and expansions
entity_expansions = {
    # Sports teams
    "team india": ["india", "indian cricket team", "men in blue"],
    "rcb": ["royal challengers bangalore", "rcb team"],
    # Tournaments
    "worldcup": ["world cup", "icc world cup", "cricket world cup"],
    "ipl": ["indian premier league", "ipl"],
    # Match formats
    "test": ["test match", "test cricket"],
    "odi": ["one day international", "50 over"],
    "t20": ["twenty20", "t20i", "t20 international"],
}

# Forbidden topics mapping by category
forbidden_topics_map = {
    "sports": ["politics", "entertainment", "technology", "business"],
    "cricket": ["football", "basketball", "tennis", "hockey"],
    "football": ["cricket", "basketball", "tennis", "hockey"],
    "basketball": ["cricket", "football", "tennis", "hockey"],
    "tennis": ["cricket", "football", "basketball", "hockey"],
    "hockey": ["cricket", "football", "basketball", "tennis"],
}

class AlertParser:
    """Parses raw user alerts into structured alert objects with enhanced metadata."""
    
    def __init__(self):
        """Initialize the alert parser with default configurations."""
        self.entity_cache = {}
        
    async def parse_alerts(self, user_alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse a list of user alerts into structured alert objects.
        
        Args:
            user_alerts: List of raw user alerts
            
        Returns:
            Dict containing parsed alerts in the alertsparse format
        """
        parsed_alerts = []
        
        for alert in user_alerts:
            try:
                parsed_alert = await self._parse_single_alert(alert)
                parsed_alerts.append(parsed_alert)
            except Exception as e:
                logger.error(f"Error parsing alert {alert.get('alert_id')}: {str(e)}")
                continue
                
        return {"alertsparse": parsed_alerts}
    
    async def _parse_single_alert(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single alert into a structured format."""
        # Extract basic alert information
        alert_id = alert.get("alert_id", str(hash(json.dumps(alert, sort_keys=True))))
        category = alert.get("main_category", "").lower()
        sub_categories = [s.lower() for s in (alert.get("sub_categories") or [])]
        followup_questions = [q.lower() for q in (alert.get("followup_questions") or [])]
        custom_question = (alert.get("custom_question") or "").lower()
        
        # Extract and normalize entities
        entities = self._extract_entities(
            category, 
            sub_categories, 
            followup_questions, 
            custom_question
        )
        
        # Parse event conditions from custom question
        event_conditions = self._parse_event_conditions(custom_question, entities)
        
        # Generate contextual query
        contextual_query = self._build_contextual_query(
            category,
            sub_categories,
            followup_questions,
            custom_question,
            entities,
            event_conditions
        )
        
        # Generate priority tags
        priority_tags = self._generate_priority_tags(
            category,
            sub_categories,
            entities,
            event_conditions
        )
        
        # Determine forbidden topics
        forbidden_topics = self._get_forbidden_topics(category, sub_categories)
        
        # Generate ranking weights
        ranking_weights = self._generate_ranking_weights(
            category,
            custom_question,
            event_conditions
        )
        
        return {
            "alert_id": alert_id,
            "category": category,
            "sub_categories": sub_categories,
            "followup_questions": followup_questions,
            "custom_question": custom_question,
            "canonical_entities": list(entities),
            "event_conditions": event_conditions,
            "contextual_query": contextual_query,
            "priority_tags": priority_tags,
            "forbidden_topics": forbidden_topics,
            "ranking_weights": ranking_weights,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _extract_entities(
        self,
        category: str,
        sub_categories: List[str],
        followup_questions: List[str],
        custom_question: str
    ) -> Set[str]:
        """Extract and normalize entities from alert data."""
        entities = set()
        
        # Add category and subcategories as entities
        if category:
            entities.add(category)
        
        # Add subcategories
        entities.update(sub_categories)
        
        # Process followup questions for entities
        for question in followup_questions:
            # Simple tokenization - can be enhanced with NER
            tokens = re.findall(r'\b\w+\b', question.lower())
            entities.update(tokens)
        
        # Process custom question for entities
        if custom_question:
            # Look for quoted phrases as important entities
            quoted_entities = re.findall(r'"(.*?)"', custom_question)
            entities.update(quoted_entities)
            
            # Add individual words as potential entities
            tokens = re.findall(r'\b\w+\b', custom_question.lower())
            entities.update(tokens)
        
        # Expand entities with synonyms and related terms
        expanded_entities = set()
        for entity in entities:
            expanded_entities.add(entity)
            # Add expanded forms of entities
            if entity in entity_expansions:
                expanded_entities.update(entity_expansions[entity])
        
        # Remove common stop words and very short entities
        stop_words = {"the", "and", "or", "but", "is", "are", "in", "on", "at", "to", "for", "a", "an"}
        filtered_entities = {
            e for e in expanded_entities 
            if len(e) > 2 and e not in stop_words
        }
        
        return filtered_entities
    
    def _parse_event_conditions(
        self, 
        custom_question: str, 
        entities: Set[str]
    ) -> List[Dict[str, str]]:
        """Parse event conditions from custom question."""
        if not custom_question:
            return []
        
        conditions = []
        
        # Check for win/loss conditions
        if any(word in custom_question for word in ["win", "won", "victory", "beat"]):
            # Find the team/player that's the subject of the win
            for entity in entities:
                if entity in custom_question:
                    conditions.append({"type": "win", "entity": entity})
                    break
        
        # Check for loss conditions
        elif any(word in custom_question for word in ["lose", "lost", "defeat", "beaten"]):
            for entity in entities:
                if entity in custom_question:
                    conditions.append({"type": "loss", "entity": entity})
                    break
        
        # Check for upcoming events
        elif any(word in custom_question for word in ["upcoming", "schedule", "when", "next"]):
            for entity in entities:
                if entity in custom_question:
                    conditions.append({"type": "upcoming", "entity": entity})
                    break
        
        # Check for score updates
        elif any(word in custom_question for word in ["score", "result", "update", "live"]):
            for entity in entities:
                if entity in custom_question:
                    conditions.append({"type": "score", "entity": entity})
                    break
        
        # If no specific condition found, add a general one
        if not conditions and entities:
            conditions.append({"type": "general", "entity": next(iter(entities))})
        
        return conditions
    
    def _build_contextual_query(
        self,
        category: str,
        sub_categories: List[str],
        followup_questions: List[str],
        custom_question: str,
        entities: Set[str],
        event_conditions: List[Dict[str, str]]
    ) -> str:
        """Build a contextual query for RAG retrieval."""
        query_terms = []
        
        # Add category and subcategories
        query_terms.append(category)
        query_terms.extend(sub_categories)
        
        # Add entities
        query_terms.extend(entities)
        
        # Add event-related keywords
        for condition in event_conditions:
            if condition["type"] in event_keywords:
                query_terms.extend(event_keywords[condition["type"]])
        
        # Add followup questions
        query_terms.extend(followup_questions)
        
        # Add custom question terms (excluding stop words)
        stop_words = {"the", "and", "or", "but", "is", "are", "in", "on", "at", "to", "for", "a", "an"}
        question_terms = [
            term for term in re.findall(r'\b\w+\b', custom_question.lower())
            if term not in stop_words
        ]
        query_terms.extend(question_terms)
        
        # Remove duplicates and join into a single query string
        unique_terms = []
        seen = set()
        for term in query_terms:
            if term not in seen and len(term) > 2:  # Filter out very short terms
                seen.add(term)
                unique_terms.append(term)
        
        return " ".join(unique_terms)
    
    def _generate_priority_tags(
        self,
        category: str,
        sub_categories: List[str],
        entities: Set[str],
        event_conditions: List[Dict[str, str]]
    ) -> List[str]:
        """Generate priority tags for the alert."""
        tags = set()
        
        # Add category and subcategories as tags
        tags.add(category)
        tags.update(sub_categories)
        
        # Add entities as tags
        tags.update(entities)
        
        # Add event types as tags
        for condition in event_conditions:
            tags.add(condition["type"])
            if "entity" in condition:
                tags.add(f"{condition['type']}_{condition['entity']}")
        
        return list(tags)
    
    def _get_forbidden_topics(
        self,
        category: str,
        sub_categories: List[str]
    ) -> List[str]:
        """Determine forbidden topics based on category and subcategories."""
        forbidden = set()
        
        # Add category-level forbidden topics
        if category in forbidden_topics_map:
            forbidden.update(forbidden_topics_map[category])
        
        # Add subcategory-level forbidden topics
        for subcat in sub_categories:
            if subcat in forbidden_topics_map:
                forbidden.update(forbidden_topics_map[subcat])
        
        return list(forbidden)
    
    def _generate_ranking_weights(
        self,
        category: str,
        custom_question: str,
        event_conditions: List[Dict[str, str]]
    ) -> Dict[str, float]:
        """Generate ranking weights for the alert."""
        # Default weights
        weights = {
            "recency_weight": 0.7,
            "popularity_weight": 0.5,
            "entity_match_weight": 0.8,
            "custom_event_weight": 0.6
        }
        
        # Adjust weights based on event conditions
        if event_conditions:
            # Higher weight for specific events
            if any(cond["type"] in ["win", "loss", "record"] for cond in event_conditions):
                weights["custom_event_weight"] = 1.0
                weights["recency_weight"] = 0.9  # More recent events are more important
            
            # For upcoming events, prioritize by date
            if any(cond["type"] == "upcoming" for cond in event_conditions):
                weights["recency_weight"] = 0.5
                weights["popularity_weight"] = 0.8  # More popular upcoming events first
        
        # Adjust for question complexity
        question_terms = len(re.findall(r'\b\w+\b', custom_question.lower()))
        if question_terms > 10:  # More specific questions get higher entity match weight
            weights["entity_match_weight"] = min(1.0, weights["entity_match_weight"] + 0.1)
        
        return weights


# Example usage
if __name__ == "__main__":
    # Example alert data
    example_alert = {
        "alert_id": "68ea3e598bc5b55c37af219d",
        "main_category": "Sports",
        "sub_categories": ["Cricket"],
        "followup_questions": ["team india", "worldcup", "test", "ODI"],
        "custom_question": "only give me my team win"
    }
    
    # Initialize parser
    parser = AlertParser()
    
    # Parse the alert
    import asyncio
    result = asyncio.run(parser.parse_alerts([example_alert]))
    
    # Print the result
    print(json.dumps(result, indent=2))
