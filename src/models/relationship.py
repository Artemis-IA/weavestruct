from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class RelationshipBase(BaseModel):
    source_id: str = Field(..., description="The ID of the source entity")
    target_id: str = Field(..., description="The ID of the target entity")
    type: str = Field(..., description="The type of the relationship")
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional properties of the relationship")

    class Config:
        json_schema_extra = {
            "example": {
                "source_id": "123e4567-e89b-12d3-a456-426614174001",
                "target_id": "789e4567-e89b-12d3-a456-426614174002",
                "type": "Partnership",
                "properties": {
                    "since": "2021-01-01",
                    "status": "active"
                }
            }
        }

class RelationshipCreate(RelationshipBase):
    """
    Model for creating a new relationship.
    Inherits from RelationshipBase and can be extended for additional fields required at creation.
    """
    pass

class Relationship(RelationshipBase):
    """
    Model representing a relationship with an ID, as returned from the database or API.
    """
    id: str = Field(..., description="The unique identifier of the relationship")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "456e4567-e89b-12d3-a456-426614174003",
                "source_id": "123e4567-e89b-12d3-a456-426614174001",
                "target_id": "789e4567-e89b-12d3-a456-426614174002",
                "type": "Partnership",
                "properties": {
                    "since": "2021-01-01",
                    "status": "active"
                }
            }
        }
