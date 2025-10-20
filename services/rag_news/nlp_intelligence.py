"""
Advanced NLP Intelligence Module
Provides sentiment analysis, entity recognition, topic modeling, and content understanding
"""

import logging
import re
from typing import Dict, List, Any, Tuple
from collections import Counter
import numpy as np

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class NLPIntelligence:
    """Advanced NLP capabilities for intelligent content analysis"""
    
    def __init__(self):
        self.sentiment_analyzer = None
        self.ner_model = None
        self.emotion_analyzer = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # Sentiment analysis
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1  # CPU
                )
                
                # Named Entity Recognition
                self.ner_model = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple",
                    device=-1
                )
                
                # Emotion detection for deeper understanding
                self.emotion_analyzer = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=-1
                )
                
                logger.info("✅ NLP Intelligence models loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Could not load all NLP models: {e}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment with confidence scores
        Returns: {"label": "POSITIVE/NEGATIVE", "score": float, "polarity": float}
        """
        if not text or not self.sentiment_analyzer:
            return {"label": "NEUTRAL", "score": 0.5, "polarity": 0.0}
        
        try:
            result = self.sentiment_analyzer(text[:512])[0]  # Limit text length
            polarity = result['score'] if result['label'] == 'POSITIVE' else -result['score']
            return {
                "label": result['label'],
                "score": result['score'],
                "polarity": polarity
            }
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return {"label": "NEUTRAL", "score": 0.5, "polarity": 0.0}
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities (PERSON, ORG, LOC, etc.)
        Returns: {"PERSON": [...], "ORG": [...], "LOC": [...], ...}
        """
        if not text or not self.ner_model:
            return {}
        
        try:
            entities_result = self.ner_model(text[:512])
            entities_dict = {}
            
            for entity in entities_result:
                entity_type = entity['entity_group']
                entity_text = entity['word']
                
                if entity_type not in entities_dict:
                    entities_dict[entity_type] = []
                entities_dict[entity_type].append(entity_text)
            
            return entities_dict
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return {}
    
    def detect_emotion(self, text: str) -> Dict[str, Any]:
        """
        Detect emotions in text (joy, sadness, anger, fear, etc.)
        Returns: {"emotion": str, "score": float}
        """
        if not text or not self.emotion_analyzer:
            return {"emotion": "neutral", "score": 0.5}
        
        try:
            result = self.emotion_analyzer(text[:512])[0]
            return {
                "emotion": result['label'],
                "score": result['score']
            }
        except Exception as e:
            logger.warning(f"Emotion detection failed: {e}")
            return {"emotion": "neutral", "score": 0.5}
    
    def extract_key_topics(self, text: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Extract key topics using TF-IDF-like approach
        Returns: [(topic, score), ...]
        """
        if not text:
            return []
        
        # Simple keyword extraction based on word frequency
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        
        # Filter common stop words
        stop_words = {
            'that', 'this', 'with', 'from', 'have', 'been', 'were', 'will',
            'said', 'would', 'could', 'about', 'there', 'their', 'which'
        }
        filtered_words = [w for w in words if w not in stop_words]
        
        # Count frequencies
        word_counts = Counter(filtered_words)
        
        # Return top N
        return word_counts.most_common(top_n)
    
    def compute_readability_score(self, text: str) -> float:
        """
        Compute readability score (0-1, higher = more readable)
        Uses simplified Flesch Reading Ease
        """
        if not text:
            return 0.5
        
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.5
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Simplified readability: shorter sentences = more readable
        # Normalize to 0-1 range
        readability = 1.0 / (1.0 + avg_sentence_length / 20.0)
        
        return min(1.0, max(0.0, readability))
    
    def analyze_content_quality(self, article: Dict) -> Dict[str, Any]:
        """
        Comprehensive content quality analysis
        Returns enriched metadata for intelligent ranking
        """
        title = article.get('title', '')
        content = article.get('content', '') or article.get('summary', '')
        text = f"{title} {content}"
        
        # Run all analyses
        sentiment = self.analyze_sentiment(text)
        entities = self.extract_entities(text)
        emotion = self.detect_emotion(text)
        topics = self.extract_key_topics(text)
        readability = self.compute_readability_score(text)
        
        # Compute quality score
        quality_factors = []
        
        # Factor 1: Entity richness (articles with more entities tend to be informative)
        entity_count = sum(len(v) for v in entities.values())
        entity_score = min(1.0, entity_count / 10.0)
        quality_factors.append(entity_score * 0.3)
        
        # Factor 2: Readability
        quality_factors.append(readability * 0.2)
        
        # Factor 3: Content length (not too short, not too long)
        word_count = len(text.split())
        length_score = 1.0 if 100 <= word_count <= 1000 else 0.5
        quality_factors.append(length_score * 0.2)
        
        # Factor 4: Sentiment clarity (strong sentiment = clear writing)
        sentiment_clarity = abs(sentiment['polarity'])
        quality_factors.append(sentiment_clarity * 0.15)
        
        # Factor 5: Emotion engagement
        quality_factors.append(emotion['score'] * 0.15)
        
        overall_quality = sum(quality_factors)
        
        return {
            "sentiment": sentiment,
            "entities": entities,
            "emotion": emotion,
            "key_topics": [t[0] for t in topics],
            "readability": readability,
            "quality_score": overall_quality,
            "entity_count": entity_count,
            "word_count": word_count
        }


# Global instance
_nlp_intelligence = None


def get_nlp_intelligence() -> NLPIntelligence | None:
    """Get or create NLP Intelligence instance"""
    global _nlp_intelligence
    if _nlp_intelligence is None:
        try:
            _nlp_intelligence = NLPIntelligence()
        except Exception as e:
            logger.warning(f"Could not initialize NLP Intelligence: {e}")
            return None
    return _nlp_intelligence


def enrich_articles_with_intelligence(articles: List[Dict]) -> List[Dict]:
    """
    Enrich articles with NLP intelligence metadata
    Adds sentiment, entities, quality scores, etc.
    """
    nlp = get_nlp_intelligence()
    if not nlp:
        logger.warning("NLP Intelligence not available, skipping enrichment")
        return articles
    
    enriched = []
    for article in articles:
        try:
            intelligence = nlp.analyze_content_quality(article)
            enriched_article = dict(article)
            enriched_article['nlp_intelligence'] = intelligence
            enriched.append(enriched_article)
        except Exception as e:
            logger.warning(f"Failed to enrich article: {e}")
            enriched.append(article)
    
    return enriched
