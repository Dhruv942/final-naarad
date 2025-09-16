from enum import Enum
from typing import Optional
from datetime import time
from pydantic import BaseModel, Field, root_validator


class NotificationType(str, Enum):
    real_time = "real_time"
    morning_digest = "morning_digest"
    evening_digest = "evening_digest"
    custom_time = "custom_time"


class NotificationPreference(BaseModel):
    type: NotificationType = Field(..., description="Type of notification preference")
    custom_time: Optional[time] = None

    @root_validator(pre=True)
    def set_fixed_times(cls, values):
        notif_type = values.get("type")
        
        if notif_type == NotificationType.morning_digest:
            values["custom_time"] = time(7, 0)   # 07:00 AM
        elif notif_type == NotificationType.evening_digest:
            values["custom_time"] = time(19, 0)  # 07:00 PM
        elif notif_type == NotificationType.real_time:
            values["custom_time"] = None
        elif notif_type == NotificationType.custom_time:
            if not values.get("custom_time"):
                raise ValueError("custom_time is required for custom_time type")
        return values

    def dict(self, *args, **kwargs):
        """Ensure time is stored as HH:MM string in DB"""
        d = super().dict(*args, **kwargs)
        if isinstance(d.get("custom_time"), time):
            d["custom_time"] = d["custom_time"].strftime("%H:%M")
        return d
