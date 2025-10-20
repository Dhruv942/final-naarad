"""
Feedback Controller for User Interaction Learning
Allows users to provide feedback on articles for personalization and model improvement
"""

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging

from services.rag_news.personalization_engine import get_personalization_engine
from services.rag_news.neural_reranker import get_neural_reranker

router = APIRouter()
logger = logging.getLogger(__name__)


class ArticleFeedback(BaseModel):
    """Model for article feedback"""
    user_id: str = Field(..., description="User identifier")
    article_id: str = Field(..., description="Article identifier or URL")
    action: str = Field(..., description="User action: click, read, like, share, dismiss")
    engagement_score: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Engagement quality 0-1")
    query: Optional[str] = Field(None, description="The query used to find this article")
    rank_position: Optional[int] = Field(None, description="Position in ranked list")
    article_data: Optional[Dict[str, Any]] = Field(None, description="Full article data for learning")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "article_id": "https://example.com/article",
                "action": "like",
                "engagement_score": 0.9,
                "query": "football news",
                "rank_position": 1,
                "article_data": {
                    "title": "Amazing match result",
                    "category": "sports",
                    "summary": "..."
                }
            }
        }


class FeedbackResponse(BaseModel):
    """Response after recording feedback"""
    success: bool
    message: str
    user_interactions: int
    recommendations: Optional[Dict[str, Any]] = None


@router.post("/article-feedback", response_model=FeedbackResponse)
async def record_article_feedback(feedback: ArticleFeedback):
    """
    Record user feedback on an article for personalization learning
    
    This endpoint allows the system to learn from user interactions:
    - Clicks, reads, likes, shares (positive signals)
    - Dismissals (negative signals)
    
    The feedback is used to:
    1. Personalize future article recommendations
    2. Improve neural reranking models
    3. Understand user preferences over time
    """
    try:
        # Validate action type
        valid_actions = ['click', 'read', 'like', 'share', 'dismiss']
        if feedback.action not in valid_actions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action. Must be one of: {', '.join(valid_actions)}"
            )
        
        # Get personalization engine
        personalization_engine = get_personalization_engine()
        if not personalization_engine:
            raise HTTPException(
                status_code=503,
                detail="Personalization engine not available"
            )
        
        # Prepare article data
        article_data = feedback.article_data or {
            'id': feedback.article_id,
            'link': feedback.article_id
        }
        
        # Record interaction for personalization
        personalization_engine.record_interaction(
            user_id=feedback.user_id,
            article=article_data,
            interaction_type=feedback.action,
            engagement_score=feedback.engagement_score
        )
        
        # Also record for neural reranker if query provided
        if feedback.query and feedback.rank_position:
            neural_reranker = get_neural_reranker()
            if neural_reranker:
                neural_reranker.record_feedback(
                    query=feedback.query,
                    article=article_data,
                    rank_position=feedback.rank_position,
                    user_action=feedback.action,
                    engagement_score=feedback.engagement_score
                )
        
        # Get user profile info
        profile = personalization_engine.get_user_profile(feedback.user_id)
        
        logger.info(f"✅ Recorded {feedback.action} feedback from user {feedback.user_id}")
        
        return FeedbackResponse(
            success=True,
            message=f"Feedback recorded successfully for {feedback.action}",
            user_interactions=profile.interaction_count
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")


@router.get("/user-recommendations/{user_id}")
async def get_user_recommendations(user_id: str):
    """
    Get personalized recommendations for a user based on their learned preferences
    
    Returns:
    - Top categories user is interested in
    - Keywords and topics of interest
    - Entities (people, organizations) user follows
    - Preferred news sources
    """
    try:
        personalization_engine = get_personalization_engine()
        if not personalization_engine:
            raise HTTPException(
                status_code=503,
                detail="Personalization engine not available"
            )
        
        profile = personalization_engine.get_user_profile(user_id)
        recommendations = personalization_engine.get_user_recommendations(user_id, top_k=10)
        
        return {
            "user_id": user_id,
            "total_interactions": profile.interaction_count,
            "last_updated": profile.last_updated.isoformat(),
            "recommendations": recommendations,
            "preferences": {
                "sentiment_preference": round(profile.sentiment_preference, 2),
                "preferred_readability": round(profile.preferred_readability, 2)
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@router.get("/ml-metrics")
async def get_ml_metrics():
    """
    Get performance metrics from ML/AI components
    
    Returns metrics about:
    - Neural reranker performance
    - User engagement statistics
    - Model learning progress
    """
    try:
        metrics = {
            "neural_reranker": {},
            "personalization": {}
        }
        
        # Neural reranker metrics
        neural_reranker = get_neural_reranker()
        if neural_reranker:
            metrics["neural_reranker"] = neural_reranker.get_performance_metrics()
        
        # Personalization metrics
        personalization_engine = get_personalization_engine()
        if personalization_engine:
            metrics["personalization"] = {
                "total_users": len(personalization_engine.user_profiles),
                "global_trends": dict(personalization_engine.global_trends)
            }
        
        return metrics
    
    except Exception as e:
        logger.error(f"Failed to get ML metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ML metrics: {str(e)}")


@router.post("/reset-user-profile/{user_id}")
async def reset_user_profile(user_id: str):
    """
    Reset a user's learned profile (for testing or user request)
    """
    try:
        personalization_engine = get_personalization_engine()
        if not personalization_engine:
            raise HTTPException(
                status_code=503,
                detail="Personalization engine not available"
            )
        
        # Remove user profile
        if user_id in personalization_engine.user_profiles:
            del personalization_engine.user_profiles[user_id]
            logger.info(f"🔄 Reset profile for user {user_id}")
            return {"success": True, "message": f"Profile reset for user {user_id}"}
        else:
            return {"success": False, "message": f"No profile found for user {user_id}"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset profile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset profile: {str(e)}")
