"""
News Pipeline Controller
API endpoints for managing the 3-stage personalized news pipeline.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import os

from core.personalized_news_pipeline import PersonalizedNewsPipeline
from schedulers.news_pipeline_scheduler import run_pipeline_manually
from db.mongo import db, alertsparse_collection, notification_queue_collection

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize pipeline (lazy loading)
_pipeline: Optional[PersonalizedNewsPipeline] = None


def get_pipeline() -> PersonalizedNewsPipeline:
    """Get or create pipeline instance."""
    global _pipeline
    if _pipeline is None:
        api_key = os.getenv("GEMINI_API_KEY", "AIzaSyAydFe00gWhKQYdoF7oKD6QALZMwnCfkus")
        _pipeline = PersonalizedNewsPipeline(
            db=db,
            gemini_api_key=api_key
        )
    return _pipeline


# ==================== REQUEST MODELS ====================

class AlertRequest(BaseModel):
    """Request model for processing a single alert."""
    alert_id: str = Field(..., description="Alert ID to process")
    user_id: str = Field(..., description="User ID")
    main_category: str = Field(..., description="Main category")
    sub_categories: List[str] = Field(default_factory=list, description="Sub-categories")
    followup_questions: List[str] = Field(default_factory=list, description="Follow-up questions")
    custom_question: str = Field(default="", description="Custom question")


class PipelineStatusResponse(BaseModel):
    """Response model for pipeline status."""
    status: str
    message: str
    stats: Optional[Dict[str, Any]] = None


# ==================== ENDPOINTS ====================

@router.post("/process-alert", response_model=Dict[str, Any])
async def process_single_alert(alert: AlertRequest):
    """
    Process a single alert through the full 3-stage pipeline.
    
    **Stages:**
    1. LLM Preference Parsing → alertsparse
    2. News Retrieval (RSS/Google/SERP)
    3. LLM Filtering & Ranking → notification_queue
    """
    try:
        logger.info(f"Processing alert: {alert.alert_id}")
        
        pipeline = get_pipeline()
        
        # Convert request to dict
        alert_data = alert.dict()
        
        # Run full pipeline
        result = await pipeline.process_alert_full_pipeline(
            alert_data=alert_data,
            store_to_queue=True
        )
        
        return {
            "success": True,
            "alert_id": alert.alert_id,
            "status": result.get("status"),
            "articles_found": len(result.get("articles", [])),
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error processing alert: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run-batch-processing")
async def run_batch_processing(background_tasks: BackgroundTasks):
    """
    Manually trigger batch processing of all active alerts.
    Runs in the background.
    """
    try:
        logger.info("Manual batch processing triggered")
        
        # Run in background
        background_tasks.add_task(run_pipeline_manually)
        
        return {
            "success": True,
            "message": "Batch processing started in background",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error triggering batch processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alertsparse/{alert_id}")
async def get_alertsparse(alert_id: str):
    """
    Get the parsed alertsparse object for an alert.
    Shows the output of Stage 1 (LLM Preference Parsing).
    """
    try:
        alertsparse = await alertsparse_collection.find_one({"alert_id": alert_id})
        
        if not alertsparse:
            raise HTTPException(status_code=404, detail="Alertsparse not found")
        
        # Convert ObjectId to string
        alertsparse["_id"] = str(alertsparse["_id"])
        
        return {
            "success": True,
            "alertsparse": alertsparse
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching alertsparse: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notification-queue/{user_id}")
async def get_notification_queue(user_id: str, limit: int = 10):
    """
    Get pending notifications for a user from the notification queue.
    Shows the final output ready to be sent.
    """
    try:
        notifications = await notification_queue_collection.find(
            {"user_id": user_id, "status": "pending"}
        ).sort("created_at", -1).limit(limit).to_list(limit)
        
        # Convert ObjectIds to strings
        for notif in notifications:
            notif["_id"] = str(notif["_id"])
        
        return {
            "success": True,
            "user_id": user_id,
            "count": len(notifications),
            "notifications": notifications
        }
        
    except Exception as e:
        logger.error(f"Error fetching notification queue: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline-stats")
async def get_pipeline_stats():
    """
    Get overall pipeline statistics.
    """
    try:
        # Count documents in various collections
        alertsparse_count = await alertsparse_collection.count_documents({})
        queue_pending = await notification_queue_collection.count_documents({"status": "pending"})
        queue_sent = await notification_queue_collection.count_documents({"status": "sent"})
        queue_failed = await notification_queue_collection.count_documents({"status": "failed"})
        
        return {
            "success": True,
            "stats": {
                "alertsparse_count": alertsparse_count,
                "notification_queue": {
                    "pending": queue_pending,
                    "sent": queue_sent,
                    "failed": queue_failed,
                    "total": queue_pending + queue_sent + queue_failed
                }
            },
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error fetching pipeline stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/notification-queue/clear/{user_id}")
async def clear_notification_queue(user_id: str):
    """
    Clear all pending notifications for a user (for testing).
    """
    try:
        result = await notification_queue_collection.delete_many(
            {"user_id": user_id, "status": "pending"}
        )
        
        return {
            "success": True,
            "deleted_count": result.deleted_count,
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Error clearing notification queue: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def pipeline_health_check():
    """
    Check if the pipeline is healthy and properly configured.
    """
    try:
        # Check if pipeline can be initialized
        pipeline = get_pipeline()
        
        return {
            "success": True,
            "status": "healthy",
            "message": "Pipeline is operational",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Pipeline health check failed: {str(e)}")
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow()
        }
