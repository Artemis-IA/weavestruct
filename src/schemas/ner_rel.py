from typing import List, Optional
from pydantic import BaseModel

class EntityBase(BaseModel):
    id: str
    type: str
    properties: Optional[dict]

class EntityCreate(BaseModel):
    id: str
    type: str
    properties: Optional[dict] = {}

class EntityUpdate(BaseModel):
    properties: Optional[dict]

class RelationshipBase(BaseModel):
    source: str
    target: str
    type: str
    properties: Optional[dict] = {}

class RelationshipCreate(RelationshipBase):
    pass

class EntityResponse(BaseModel):
    id: str
    type: str
    properties: dict

class RelationshipResponse(BaseModel):
    source: str
    target: str
    type: str
    properties: dict

class EntityRelationshipsResponse(BaseModel):
    entity: EntityResponse
    relationships: List[RelationshipResponse]
