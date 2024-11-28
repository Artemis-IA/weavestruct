from pydantic import BaseModel
from typing import Dict, Any

class Document(BaseModel):
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str

    class Config:
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "title": "Sample Document",
                "content": "This is the content of the document.",
                "metadata": {
                    "author": "John Doe",
                    "tags": ["example", "sample"]
                },
                "created_at": "2023-11-12T10:00:00Z",
                "updated_at": "2023-11-12T12:00:00Z"
            }
        }
