from fastapi import APIRouter
from models.alerts import AlertCreate, AlertResponse
from controllers.alerts_controller import (
    create_alert,
    get_alerts_by_user,
    update_alert_by_id,
    delete_alert_by_id
    
)
from typing import List

router = APIRouter()


@router.post("/", response_model=AlertResponse)
async def post_alert(alert: AlertCreate):
    return await create_alert(alert)


@router.get("/{user_id}", response_model=List[AlertResponse])
async def get_alerts(user_id: str):
    return await get_alerts_by_user(user_id)


@router.put("/{user_id}/{alert_id}", response_model=AlertResponse)
async def put_alert_by_id(user_id: str, alert_id: str, update_data: dict):
    """
    Update specific alert by user_id + alert_id.
    Example update_data:
    {
      "sub_categories": ["T20", "Notify me if RCB wins"],
      "is_active": false
    }
    """
    return await update_alert_by_id(user_id, alert_id, update_data)

@router.delete("/{user_id}/{alert_id}")
async def delete_alert(user_id: str, alert_id: str):
    return await delete_alert_by_id(user_id, alert_id)  

