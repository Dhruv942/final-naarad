"""
Neural Reranking System with Learning from Feedback
Uses cross-encoder models and learns from user interactions
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import deque
from datetime import datetime

try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False

logger = logging.getLogger(__name__)


class NeuralReranker:
    """
    Advanced neural reranker with feedback learning
    Combines cross-encoder scores with learned user preferences
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = None
        self.model_name = model_name
        
        # Feedback storage for continuous learning
        self.feedback_queue = deque(maxlen=1000)  # Store recent feedback
        self.query_performance = {}  # Track query -> performance metrics
        
        # Feature importance weights (learned over time)
        self.feature_weights = {
            'cross_encoder_score': 0.40,
            'semantic_similarity': 0.20,
            'nlp_quality': 0.15,
            'personalization': 0.15,
            'freshness': 0.10
        }
        
        if CROSSENCODER_AVAILABLE:
            try:
                self.model = CrossEncoder(model_name, max_length=512)
                logger.info(f"✅ Neural reranker loaded: {model_name}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load cross-encoder: {e}")
    
    def rerank_with_intelligence(
        self,
        query: str,
        articles: List[Dict],
        top_k: int = 10,
        use_feedback: bool = True
    ) -> List[Dict]:
        """
        Rerank articles using neural model + learned preferences
        
        Args:
            query: Search query
            articles: Articles to rerank
            top_k: Number of top results to return
            use_feedback: Whether to incorporate feedback learning
        
        Returns:
            Reranked articles with scores
        """
        if not articles:
            return []
        
        # Step 1: Get cross-encoder scores
        ce_scores = self._get_cross_encoder_scores(query, articles)
        
        # Step 2: Extract additional features
        features_matrix = self._extract_features(articles, ce_scores)
        
        # Step 3: Combine scores with learned weights
        final_scores = self._compute_weighted_scores(features_matrix, use_feedback)
        
        # Step 4: Attach scores and sort
        reranked = []
        for i, article in enumerate(articles):
            article_copy = dict(article)
            article_copy['neural_rerank_score'] = float(final_scores[i])
            article_copy['ce_score'] = float(ce_scores[i])
            article_copy['feature_breakdown'] = {
                feat: float(features_matrix[i][j])
                for j, feat in enumerate(self.feature_weights.keys())
            }
            reranked.append(article_copy)
        
        # Sort by final score
        reranked.sort(key=lambda x: x['neural_rerank_score'], reverse=True)
        
        logger.info(f"🧠 Neural reranking: {len(articles)} -> top {min(top_k, len(reranked))}")
        return reranked[:top_k]
    
    def _get_cross_encoder_scores(self, query: str, articles: List[Dict]) -> np.ndarray:
        """Get scores from cross-encoder model"""
        if not self.model:
            # Fallback: use existing scores or return neutral
            scores = []
            for article in articles:
                score = article.get('rerank_score', article.get('retrieve_score', 0.5))
                scores.append(score)
            return np.array(scores)
        
        try:
            # Prepare query-document pairs
            pairs = []
            for article in articles:
                title = article.get('title', '')
                summary = article.get('summary', '') or article.get('content', '')[:200]
                doc_text = f"{title}. {summary}"
                pairs.append([query, doc_text])
            
            # Get scores from model
            scores = self.model.predict(pairs)
            
            # Normalize to 0-1 range using sigmoid
            normalized_scores = 1.0 / (1.0 + np.exp(-np.array(scores)))
            
            return normalized_scores
        except Exception as e:
            logger.warning(f"Cross-encoder scoring failed: {e}")
            return np.array([0.5] * len(articles))
    
    def _extract_features(self, articles: List[Dict], ce_scores: np.ndarray) -> np.ndarray:
        """
        Extract feature matrix for all articles
        Features: [ce_score, semantic_sim, nlp_quality, personalization, freshness]
        """
        n_articles = len(articles)
        n_features = len(self.feature_weights)
        features = np.zeros((n_articles, n_features))
        
        for i, article in enumerate(articles):
            # Feature 0: Cross-encoder score
            features[i, 0] = ce_scores[i]
            
            # Feature 1: Semantic similarity (from retrieval)
            features[i, 1] = article.get('retrieve_score', 0.5)
            
            # Feature 2: NLP quality score
            if 'nlp_intelligence' in article:
                features[i, 2] = article['nlp_intelligence'].get('quality_score', 0.5)
            else:
                features[i, 2] = 0.5
            
            # Feature 3: Personalization score
            features[i, 3] = article.get('personalization_score', 0.5)
            
            # Feature 4: Freshness weight
            features[i, 4] = article.get('fresh_weight', 0.5)
        
        return features
    
    def _compute_weighted_scores(
        self,
        features: np.ndarray,
        use_feedback: bool = True
    ) -> np.ndarray:
        """
        Compute final scores using learned feature weights
        """
        # Get current weights as array
        weights = np.array([
            self.feature_weights['cross_encoder_score'],
            self.feature_weights['semantic_similarity'],
            self.feature_weights['nlp_quality'],
            self.feature_weights['personalization'],
            self.feature_weights['freshness']
        ])
        
        # Weighted sum
        scores = np.dot(features, weights)
        
        # Normalize to 0-1 range
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        return scores
    
    def record_feedback(
        self,
        query: str,
        article: Dict,
        rank_position: int,
        user_action: str,
        engagement_score: float
    ):
        """
        Record feedback for learning
        
        Args:
            query: The query that was used
            article: Article that was ranked
            rank_position: Position in ranked list (1-indexed)
            user_action: 'click', 'read', 'like', 'share', 'dismiss'
            engagement_score: 0-1 score of engagement quality
        """
        feedback_entry = {
            'timestamp': datetime.now(),
            'query': query,
            'article_id': article.get('id', '') or article.get('link', ''),
            'rank_position': rank_position,
            'user_action': user_action,
            'engagement_score': engagement_score,
            'features': article.get('feature_breakdown', {})
        }
        
        self.feedback_queue.append(feedback_entry)
        
        # Update query performance metrics
        if query not in self.query_performance:
            self.query_performance[query] = {
                'total_interactions': 0,
                'positive_interactions': 0,
                'avg_rank_clicked': 0.0
            }
        
        qp = self.query_performance[query]
        qp['total_interactions'] += 1
        
        if user_action in ['click', 'read', 'like', 'share']:
            qp['positive_interactions'] += 1
            qp['avg_rank_clicked'] = (
                (qp['avg_rank_clicked'] * (qp['positive_interactions'] - 1) + rank_position)
                / qp['positive_interactions']
            )
        
        # Trigger learning if enough feedback accumulated
        if len(self.feedback_queue) >= 100 and len(self.feedback_queue) % 50 == 0:
            self._update_feature_weights()
    
    def _update_feature_weights(self):
        """
        Update feature weights based on feedback (simple online learning)
        Uses gradient-free optimization based on correlation with engagement
        """
        if len(self.feedback_queue) < 50:
            return
        
        try:
            # Collect feature values and engagement scores
            feature_names = list(self.feature_weights.keys())
            feature_values = {feat: [] for feat in feature_names}
            engagement_scores = []
            
            for feedback in self.feedback_queue:
                features = feedback.get('features', {})
                engagement = feedback['engagement_score']
                
                # Map user actions to engagement multipliers
                action_multipliers = {
                    'dismiss': 0.0,
                    'click': 0.5,
                    'read': 0.7,
                    'like': 0.9,
                    'share': 1.0
                }
                multiplier = action_multipliers.get(feedback['user_action'], 0.5)
                adjusted_engagement = engagement * multiplier
                
                engagement_scores.append(adjusted_engagement)
                
                for feat in feature_names:
                    feature_values[feat].append(features.get(feat, 0.5))
            
            # Calculate correlation between each feature and engagement
            engagement_array = np.array(engagement_scores)
            correlations = {}
            
            for feat in feature_names:
                feat_array = np.array(feature_values[feat])
                if len(feat_array) > 0 and feat_array.std() > 0:
                    corr = np.corrcoef(feat_array, engagement_array)[0, 1]
                    correlations[feat] = max(0.0, corr)  # Only positive correlations
                else:
                    correlations[feat] = 0.5
            
            # Update weights based on correlations (with smoothing)
            learning_rate = 0.1
            total_corr = sum(correlations.values()) or 1.0
            
            for feat in feature_names:
                target_weight = correlations[feat] / total_corr
                current_weight = self.feature_weights[feat]
                new_weight = current_weight * (1 - learning_rate) + target_weight * learning_rate
                self.feature_weights[feat] = new_weight
            
            # Normalize weights to sum to 1.0
            total_weight = sum(self.feature_weights.values())
            for feat in feature_names:
                self.feature_weights[feat] /= total_weight
            
            logger.info(f"📈 Updated neural reranker weights: {self.feature_weights}")
        
        except Exception as e:
            logger.warning(f"Failed to update feature weights: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        if not self.feedback_queue:
            return {
                'total_feedback': 0,
                'avg_engagement': 0.0,
                'feature_weights': self.feature_weights
            }
        
        total_feedback = len(self.feedback_queue)
        avg_engagement = np.mean([f['engagement_score'] for f in self.feedback_queue])
        
        action_distribution = {}
        for feedback in self.feedback_queue:
            action = feedback['user_action']
            action_distribution[action] = action_distribution.get(action, 0) + 1
        
        return {
            'total_feedback': total_feedback,
            'avg_engagement': float(avg_engagement),
            'feature_weights': dict(self.feature_weights),
            'action_distribution': action_distribution,
            'queries_tracked': len(self.query_performance)
        }


# Global instance
_neural_reranker = None


def get_neural_reranker() -> NeuralReranker | None:
    """Get or create neural reranker instance"""
    global _neural_reranker
    if _neural_reranker is None:
        try:
            _neural_reranker = NeuralReranker()
        except Exception as e:
            logger.warning(f"Could not initialize neural reranker: {e}")
            return None
    return _neural_reranker
