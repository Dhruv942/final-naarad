from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid


class CategoryType(str, Enum):
    Sports = "Sports"
    News = "News"
    Movies = "Movies"
    YouTube = "YouTube"
    Custom_Input = "Custom_Input"


class FrequencyType(str, Enum):
    realtime = "realtime"
    hourly = "hourly"
    daily = "daily"
    weekly = "weekly"


class ScheduleSettings(BaseModel):
    frequency: FrequencyType = Field(..., description="How often to send notifications")
    time: Optional[str] = Field(default="09:00", description="Time for daily/weekly alerts (HH:MM format)")
    days: Optional[List[str]] = Field(default=[], description="Days for weekly alerts: ['monday', 'friday']")
    timezone: Optional[str] = Field(default="Asia/Kolkata", description="User timezone")


class AlertBase(BaseModel):
    main_category: CategoryType = Field(..., description="Fixed main category enum")
    sub_categories: Optional[List[str]] = Field(
        default=None, description="Frontend selected sub-categories"
    )
    followup_questions: Optional[List[str]] = Field(
        default=None, description="Follow-up questions from frontend"
    )
    custom_question: Optional[str] = Field(
        default=None, description="Custom question text from frontend"
    )


class AlertCreate(AlertBase):
    user_id: str = Field(..., description="User ID from users collection")


class AlertResponse(AlertBase):
    alert_id: str
    user_id: str
    is_active: bool

    class Config:
        use_enum_values = True
        populate_by_name = True
