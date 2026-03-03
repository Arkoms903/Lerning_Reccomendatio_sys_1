from pydantic import BaseModel, Field, validator
from typing import Optional, List


# -----------------------------------------
# Request Models
# -----------------------------------------

class RecommendRequest(BaseModel):
    """For existing users already in the system."""
    user_id: str = Field(..., example="U050", description="Existing user ID")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of recommendations to return")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "U050",
                "top_k": 5
            }
        }


class NewUserRecommendRequest(BaseModel):
    """
    For brand new users not in the system.
    They provide the list of topic_ids they have already completed
    and optionally a display name.
    """
    user_id: Optional[str] = Field(
        default="new_user",
        example="new_user_john",
        description="Any label for the new user (not saved to DB)"
    )
    completed_topic_ids: List[str] = Field(
        ...,
        example=["T001", "T002", "T003"],
        description="List of topic IDs the user has already completed"
    )
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of recommendations to return")

    @validator("completed_topic_ids")
    def must_not_be_empty(cls, v):
        if len(v) == 0:
            raise ValueError(
                "completed_topic_ids cannot be empty. "
                "Use GET /recommend/cold-start for users with zero history."
            )
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "new_user_john",
                "completed_topic_ids": ["T001", "T002", "T003"],
                "top_k": 5
            }
        }


# -----------------------------------------
# Response Models
# -----------------------------------------

class TopicRecommendation(BaseModel):
    topic_id: str = Field(..., example="T021")
    topic_name: str = Field(..., example="OOP Concepts")
    difficulty: str = Field(..., example="Intermediate")
    estimated_hours: Optional[str] = Field(None, example="8")
    score: Optional[float] = Field(None, example=0.204)


class RecommendResponse(BaseModel):
    user_id: str = Field(..., example="U050")
    total: int = Field(..., example=5)
    is_cold_start: bool = Field(..., description="True if user had no history")
    is_new_user: bool = Field(default=False, description="True if this was a new user recommendation")
    recommendations: List[TopicRecommendation]


class CompletedTopicsResponse(BaseModel):
    user_id: str
    total_completed: int
    completed_topic_ids: List[str]


class HealthResponse(BaseModel):
    status: str = "ok"
    message: str = "Learning Path Recommender API is running"