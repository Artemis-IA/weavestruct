# src/schemas/inference.py
from pydantic import BaseModel
from fastapi import Form
from typing import List, Dict
from datetime import datetime

class InferenceRequest(BaseModel):
    labels: List[str]
    threshold: float
    flat_ner: bool
    multi_label: bool
    batch_size: int

    @classmethod
    def as_form(
        cls,
        labels: str = Form("PERSON,PLACE,THING,ORGANIZATION,DATE,TIME", description="Types d'entités à extraire"),
        threshold: float = Form(0.3, description="Seuil de confiance pour l'inférence"),
        flat_ner: bool = Form(True, description="If need to extract parts of complex entities: False"),
        multi_label: bool = Form(False, description="If entities belong to several classes: True"),
        batch_size: int = Form(12, description="Taille du lot d'inférence")
    ) -> "InferenceRequest":
        # Les labels sont séparés par des virgules dans le formulaire, donc nous les convertissons en liste
        return cls(
            labels=labels.split(","),
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
            batch_size=batch_size
        )

class InferenceResponse(BaseModel):
    id: int
    file_path: str
    entities: List[Dict]
    created_at: datetime

    class Config:   
        from_attributes = True
