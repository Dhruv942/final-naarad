"""
Personalization Engine with User Behavior Learning
Learns from user interactions and provides personalized ranking
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class UserProfile:
    """Represents a user's learned preferences and behavior patterns"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        
        # Interest vectors
        self.category_preferences = defaultdict(float)  # category -> preference score
        self.keyword_interests = defaultdict(float)  # keyword -> interest score
        self.entity_interests = defaultdict(float)  # entity -> interest score
        self.source_preferences = defaultdict(float)  # source domain -> preference
        
        # Behavioral patterns
        self.reading_times = defaultdict(list)  # category -> [reading_times]
        self.engagement_history = []  # List of (timestamp, article_id, engagement_score)
        
        # Temporal preferences
        self.time_of_day_preferences = defaultdict(float)  # hour -> preference
        self.day_of_week_preferences = defaultdict(float)  # weekday -> preference
        
        # Sentiment and emotion preferences
        self.sentiment_preference = 0.0  # -1 (negative) to 1 (positive)
        self.emotion_preferences = defaultdict(float)  # emotion -> preference
        
        # Quality preferences
        self.preferred_readability = 0.5  # 0-1
        self.preferred_length = 500  # words
        
        # Metadata
        self.last_updated = datetime.now()
        self.interaction_count = 0
    
    def update_from_feedback(self, article: Dict, engagement_score: float, interaction_type: str):
        """
        Update profile based on user feedback
        
        Args:
            article: Article that user interacted with
            engagement_score: 0-1 score (0=negative, 0.5=neutral, 1=positive)
            interaction_type: 'read', 'like', 'share', 'dismiss', 'click'
        """
        self.interaction_count += 1
        self.last_updated = datetime.now()
        
        # Weight by interaction type
        weights = {
            'dismiss': -0.5,
            'click': 0.3,
            'read': 0.5,
            'like': 1.0,
            'share': 1.5
        }
        weight = weights.get(interaction_type, 0.5)
        weighted_score = engagement_score * weight
        
        # Update category preferences
        category = article.get('category', '').lower()
        if category:
            self.category_preferences[category] += weighted_score * 0.1
        
        # Update keyword interests
        title = article.get('title', '').lower()
        keywords = title.split()
        for keyword in keywords:
            if len(keyword) > 3:
                self.keyword_interests[keyword] += weighted_score * 0.05
        
        # Update entity interests (if available)
        if 'nlp_intelligence' in article:
            entities = article['nlp_intelligence'].get('entities', {})
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    self.entity_interests[entity.lower()] += weighted_score * 0.08
        
        # Update source preferences
        source = article.get('source', '') or article.get('link', '')
        if source:
            from urllib.parse import urlparse
            domain = urlparse(source).netloc
            if domain:
                self.source_preferences[domain] += weighted_score * 0.05
        
        # Update sentiment preference
        if 'nlp_intelligence' in article:
            sentiment = article['nlp_intelligence'].get('sentiment', {})
            polarity = sentiment.get('polarity', 0.0)
            self.sentiment_preference += polarity * weighted_score * 0.05
            
            # Update emotion preferences
            emotion = article['nlp_intelligence'].get('emotion', {})
            emotion_label = emotion.get('emotion', 'neutral')
            self.emotion_preferences[emotion_label] += weighted_score * 0.05
        
        # Update temporal preferences
        now = datetime.now()
        hour = now.hour
        weekday = now.weekday()
        self.time_of_day_preferences[hour] += weighted_score * 0.03
        self.day_of_week_preferences[weekday] += weighted_score * 0.03
        
        # Store engagement history
        article_id = article.get('id', '') or article.get('link', '')
        self.engagement_history.append((datetime.now(), article_id, weighted_score))
        
        # Keep only last 1000 interactions
        if len(self.engagement_history) > 1000:
            self.engagement_history = self.engagement_history[-1000:]
        
        # Normalize preferences periodically
        if self.interaction_count % 50 == 0:
            self._normalize_preferences()
    
    def _normalize_preferences(self):
        """Normalize preference scores to prevent unbounded growth"""
        # Apply decay to old preferences
        decay_factor = 0.95
        
        for key in self.category_preferences:
            self.category_preferences[key] *= decay_factor
        
        for key in self.keyword_interests:
            self.keyword_interests[key] *= decay_factor
        
        for key in self.entity_interests:
            self.entity_interests[key] *= decay_factor
        
        self.sentiment_preference *= decay_factor
    
    def get_personalization_score(self, article: Dict) -> float:
        """
        Calculate personalized relevance score for an article
        Returns: 0-1 score based on user preferences
        """
        score = 0.0
        factors = []
        
        # Category match
        category = article.get('category', '').lower()
        if category and category in self.category_preferences:
            cat_score = self.category_preferences[category]
            factors.append(('category', cat_score * 0.25))
        
        # Keyword matches
        title = article.get('title', '').lower()
        keyword_score = 0.0
        keywords = title.split()
        matches = 0
        for keyword in keywords:
            if keyword in self.keyword_interests:
                keyword_score += self.keyword_interests[keyword]
                matches += 1
        if matches > 0:
            keyword_score = keyword_score / matches
            factors.append(('keywords', keyword_score * 0.20))
        
        # Entity matches
        if 'nlp_intelligence' in article:
            entities = article['nlp_intelligence'].get('entities', {})
            entity_score = 0.0
            entity_matches = 0
            for entity_list in entities.values():
                for entity in entity_list:
                    if entity.lower() in self.entity_interests:
                        entity_score += self.entity_interests[entity.lower()]
                        entity_matches += 1
            if entity_matches > 0:
                entity_score = entity_score / entity_matches
                factors.append(('entities', entity_score * 0.15))
        
        # Source preference
        source = article.get('source', '') or article.get('link', '')
        if source:
            from urllib.parse import urlparse
            domain = urlparse(source).netloc
            if domain and domain in self.source_preferences:
                source_score = self.source_preferences[domain]
                factors.append(('source', source_score * 0.10))
        
        # Sentiment alignment
        if 'nlp_intelligence' in article:
            sentiment = article['nlp_intelligence'].get('sentiment', {})
            polarity = sentiment.get('polarity', 0.0)
            # Preference alignment: prefer articles that match user's sentiment preference
            sentiment_alignment = 1.0 - abs(polarity - self.sentiment_preference)
            factors.append(('sentiment', sentiment_alignment * 0.10))
            
            # Emotion preference
            emotion = article['nlp_intelligence'].get('emotion', {})
            emotion_label = emotion.get('emotion', 'neutral')
            if emotion_label in self.emotion_preferences:
                emotion_score = self.emotion_preferences[emotion_label]
                factors.append(('emotion', emotion_score * 0.10))
        
        # Temporal preference (time of day)
        now = datetime.now()
        hour = now.hour
        if hour in self.time_of_day_preferences:
            temporal_score = self.time_of_day_preferences[hour]
            factors.append(('temporal', temporal_score * 0.05))
        
        # Sum all factors
        score = sum(factor[1] for factor in factors)
        
        # Normalize to 0-1 range using sigmoid
        normalized_score = 1.0 / (1.0 + np.exp(-score))
        
        return normalized_score


class PersonalizationEngine:
    """Manages user profiles and provides personalized ranking"""
    
    def __init__(self):
        self.user_profiles: Dict[str, UserProfile] = {}
        self.global_trends = defaultdict(float)  # Track global popularity
        logger.info("✅ Personalization Engine initialized")
    
    def get_user_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id)
        return self.user_profiles[user_id]
    
    def record_interaction(
        self,
        user_id: str,
        article: Dict,
        interaction_type: str,
        engagement_score: float = 0.5
    ):
        """
        Record user interaction for learning
        
        Args:
            user_id: User identifier
            article: Article that was interacted with
            interaction_type: 'read', 'like', 'share', 'dismiss', 'click'
            engagement_score: 0-1 score of engagement quality
        """
        profile = self.get_user_profile(user_id)
        profile.update_from_feedback(article, engagement_score, interaction_type)
        
        # Update global trends
        category = article.get('category', '').lower()
        if category:
            self.global_trends[category] += 0.01
        
        logger.info(f"📊 Recorded {interaction_type} interaction for user {user_id}")
    
    def personalize_articles(
        self,
        user_id: str,
        articles: List[Dict],
        blend_factor: float = 0.7
    ) -> List[Dict]:
        """
        Personalize article ranking based on user profile
        
        Args:
            user_id: User identifier
            articles: List of articles to personalize
            blend_factor: 0-1, how much to blend personalization (1=full personalization)
        
        Returns:
            Articles with added personalization scores
        """
        if not articles:
            return articles
        
        profile = self.get_user_profile(user_id)
        
        # Skip personalization for new users (cold start)
        if profile.interaction_count < 3:
            logger.info(f"⚡ Cold start for user {user_id}, using default ranking")
            for article in articles:
                article['personalization_score'] = 0.5
            return articles
        
        personalized = []
        for article in articles:
            article_copy = dict(article)
            
            # Calculate personalization score
            personal_score = profile.get_personalization_score(article)
            
            # Blend with existing score (if any)
            existing_score = article.get('preference_core', 0.5)
            blended_score = (blend_factor * personal_score) + ((1 - blend_factor) * existing_score)
            
            article_copy['personalization_score'] = personal_score
            article_copy['blended_score'] = blended_score
            
            personalized.append(article_copy)
        
        # Sort by blended score
        personalized.sort(key=lambda x: x.get('blended_score', 0.0), reverse=True)
        
        logger.info(f"🎯 Personalized {len(articles)} articles for user {user_id}")
        return personalized
    
    def get_user_recommendations(
        self,
        user_id: str,
        category: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[str, List[str]]:
        """
        Get personalized recommendations for user
        
        Returns:
            Dictionary with recommended keywords, entities, sources
        """
        profile = self.get_user_profile(user_id)
        
        recommendations = {
            'categories': [],
            'keywords': [],
            'entities': [],
            'sources': []
        }
        
        # Top categories
        sorted_categories = sorted(
            profile.category_preferences.items(),
            key=lambda x: x[1],
            reverse=True
        )
        recommendations['categories'] = [cat for cat, _ in sorted_categories[:top_k]]
        
        # Top keywords
        sorted_keywords = sorted(
            profile.keyword_interests.items(),
            key=lambda x: x[1],
            reverse=True
        )
        recommendations['keywords'] = [kw for kw, _ in sorted_keywords[:top_k]]
        
        # Top entities
        sorted_entities = sorted(
            profile.entity_interests.items(),
            key=lambda x: x[1],
            reverse=True
        )
        recommendations['entities'] = [ent for ent, _ in sorted_entities[:top_k]]
        
        # Top sources
        sorted_sources = sorted(
            profile.source_preferences.items(),
            key=lambda x: x[1],
            reverse=True
        )
        recommendations['sources'] = [src for src, _ in sorted_sources[:top_k]]
        
        return recommendations


# Global instance
_personalization_engine = None


def get_personalization_engine() -> PersonalizationEngine:
    """Get or create personalization engine instance"""
    global _personalization_engine
    if _personalization_engine is None:
        _personalization_engine = PersonalizationEngine()
    return _personalization_engine
