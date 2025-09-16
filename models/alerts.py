from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from bson import ObjectId


class CategoryType(str, Enum):
    Sports = "Sports"
    News = "News"
    Movies = "Movies"
    YouTube = "YouTube"
    Custom_Input = "Custom_Input"


class AlertBase(BaseModel):
    main_category: CategoryType = Field(..., description="Fixed main category enum")
    sub_categories: Optional[List[str]] = Field(
        default=None, description="Frontend selected sub-categories"
    )


class AlertCreate(AlertBase):
    user_id: str = Field(..., description="User ID from users collection")


class AlertResponse(AlertBase):
    id: str = Field(..., alias="_id")
    user_id: str
    is_active: bool

    class Config:
        use_enum_values = True
        populate_by_name = True
        json_encoders = {ObjectId: str}
