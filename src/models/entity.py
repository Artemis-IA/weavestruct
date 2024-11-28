from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class EntityBase(BaseModel):
    name: str
    type: str
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Sample Entity",
                "type": "Organization",
                "properties": {
                    "location": "New York",
                    "employees": 100
                }
            }
        }

class EntityCreate(EntityBase):
    """
    Model for creating a new entity.
    Inherits from EntityBase and can be extended for additional fields required at creation.
    """
    pass

class Entity(EntityBase):
    """
    Model representing an entity with an ID, as returned from the database or API.
    """
    id: str = Field(..., description="The unique identifier of the entity")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174001",
                "name": "Sample Entity",
                "type": "Organization",
                "properties": {
                    "location": "New York",
                    "employees": 100
                }
            }
        }
