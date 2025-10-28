"""
Feedback Controller for User Interaction Learning
Handles user feedback on articles and recommendations
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
from datetime import datetime, timezone

from db.mongo import db

router = APIRouter()
logger = logging.getLogger(__name__)

class ArticleFeedback(BaseModel):
    """Model for article feedback"""
    user_id: str = Field(..., description="User identifier")
    article_id: str = Field(..., description="Article identifier or URL")
    alert_id: Optional[str] = Field(None, description="Associated alert ID if applicable")
    interaction_type: str = Field(
        ...,
        description="Type of interaction (click, like, share, dismiss, read, save)"
    )
    rating: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Optional 1-5 star rating (5=best)"
    )
    feedback_text: Optional[str] = Field(
        None,
        description="Optional free-form text feedback"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata about the interaction"
    )

class FeedbackResponse(BaseModel):
    """Response after recording feedback"""
    success: bool
    message: str
    interaction_id: Optional[str] = None

async def _record_interaction(
    user_id: str,
    article_id: str,
    interaction_type: str,
    alert_id: Optional[str] = None,
    rating: Optional[int] = None,
    feedback_text: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Record user interaction with an article"""
    try:
        collection = db.get_collection("user_interactions")
        
        interaction = {
            "user_id": user_id,
            "article_id": article_id,
            "alert_id": alert_id,
            "interaction_type": interaction_type,
            "rating": rating,
            "feedback_text": feedback_text,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        
        result = await collection.insert_one(interaction)
        return str(result.inserted_id)
        
    except Exception as e:
        logger.error(f"Error recording interaction: {e}")
        raise

@router.post("/feedback", response_model=FeedbackResponse)
async def record_article_feedback(feedback: ArticleFeedback):
    """
    Record user feedback on an article
    
    This endpoint records various types of user interactions:
    - Clicks, reads, likes, shares (positive signals)
    - Dismissals (negative signals)
    - Ratings and feedback text
    """
    try:
        interaction_id = await _record_interaction(
            user_id=feedback.user_id,
            article_id=feedback.article_id,
            alert_id=feedback.alert_id,
            interaction_type=feedback.interaction_type,
            rating=feedback.rating,
            feedback_text=feedback.feedback_text,
            metadata=feedback.metadata
        )
        
        return FeedbackResponse(
            success=True,
            message="Feedback recorded successfully",
            interaction_id=interaction_id
        )
        
    except Exception as e:
        error_msg = f"Failed to record feedback: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


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
