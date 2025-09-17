from fastapi import APIRouter
from typing import List
from models.alerts import AlertCreate, AlertResponse
from controllers.alerts_controller import (
    create_alert,
    get_alerts_by_user,
    update_alert_by_id,
    delete_alert_by_id,
    paused_alert_by_id,
)

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.post("/", response_model=AlertResponse)
async def post_alert(alert: AlertCreate):
    """Create a new alert"""
    return await create_alert(alert)


@router.get("/{user_id}", response_model=List[AlertResponse])
async def get_alerts(user_id: str):
    """Get all alerts for a user"""
    return await get_alerts_by_user(user_id)


@router.put("/{user_id}/{alert_id}", response_model=AlertResponse)
async def put_alert_by_id(user_id: str, alert_id: str, update_data: dict):
    """
    Update specific alert by user_id + alert_id.
    Example body:
    {
      "sub_categories": ["T20", "Notify me if RCB wins"],
      "is_active": false
    }
    """
    return await update_alert_by_id(user_id, alert_id, update_data)


@router.delete("/{user_id}/{alert_id}")
async def delete_alert(user_id: str, alert_id: str):
    """Delete an alert"""
    return await delete_alert_by_id(user_id, alert_id)


@router.put("/{user_id}/{alert_id}/pause", response_model=AlertResponse)
async def pause_alert(user_id: str, alert_id: str):
    """Pause an alert (set is_active = False)"""
    return await paused_alert_by_id(user_id, alert_id)
