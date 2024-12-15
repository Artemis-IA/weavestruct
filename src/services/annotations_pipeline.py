import os
import types
from typing import List, Optional, Union, Dict

import torch
from datasets import Dataset
import spacy
from spacy.tokens import Doc
from spacy.util import filter_spans, minibatch
from spacy.language import Language
from loguru import logger

from glirel import GLiREL
from glirel.spacy_integration import SpacyGLiRELWrapper
from glirel.modules.utils import constrain_relations_by_entity_type
from gliner import GLiNER
from gliner_spacy.pipeline import GlinerSpacy
from transformers import AutoModelForTokenClassification, pipeline
from src.config import Settings


@Language.factory("gliner_spacy_factory")
def create_gliner_spacy(nlp: Language, name: str, gliner_model: str, device: str) -> GlinerSpacy:
    return GlinerSpacy(
        nlp=nlp,
        name=name,
        gliner_model=gliner_model,
        chunk_size=250,
        labels=["person", "organization", "email"],
        style="ent",
        threshold=0.3,
        map_location=device,
    )
@Language.factory("glirel_factory")
def create_glirel(nlp: Language, name: str, glirel_model: str, device: str) -> SpacyGLiRELWrapper:
    return SpacyGLiRELWrapper(
        pretrained_model_name_or_path=glirel_model,
        device=device,
        threshold=0.3,
    )

class AnnotationPipelines:
    def __init__(self):
        self.settings = Settings()
        self.nlp = self._initialize_spacy_pipeline()

    def _initialize_spacy_pipeline(self):
        nlp = spacy.blank("fr")
        config = self.settings.get_spacy_config()
        # Ajouter GLiNER
        nlp.add_pipe(
            "gliner_spacy_factory",
            name="gliner_spacy",
            config={"gliner_model": config["gliner_model"], "device": config["device"]},
        )
        # Ajouter GLiREL
        nlp.add_pipe(
            "glirel_factory",
            name="glirel",
            after="gliner_spacy",
            config={"glirel_model": config["glirel_model"], "device": config["device"]},
        )
        return nlp

    def annotate_ner(self, text: str, labels: List[str]) -> List[Dict]:
        if not text.strip():
            logger.warning("Empty or whitespace-only text provided for NER annotation.")
            return []

        try:
            # Pass text with labels as context
            doc = self.nlp.pipe([(text, {'re_labels': labels})], as_tuples=True)
            doc = list(doc)[0]  # Retrieve the first document
            filtered_ents = filter_spans(doc.ents)
            entities = [
                {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
                for ent in filtered_ents
                if isinstance(ent.start_char, int) and isinstance(ent.end_char, int) and ent.start_char < ent.end_char
            ]
            logger.info(f"Extracted entities: {entities}")
            return entities
        except Exception as e:
            logger.error(f"Error during NER annotation: {e}")
            return []

    def extract_entities(self, texts: List[str], labels: Optional[List[str]] = None) -> List[List[Dict]]:
        if not texts:
            logger.error("No texts provided for entity extraction.")
            return []
        if labels is None:
            logger.warning("No labels provided. Defaulting to ['person', 'organization', 'email'].")
            labels = ["person", "organization", "email"]

        try:
            # Ensure input adheres to required format
            annotated_texts = []
            for text in texts:
                if text.strip():
                    if isinstance(labels, dict):
                        labels_and_constraints = labels  # Initialisez `labels_and_constraints`
                        labels = list(labels.keys())  # Extraire uniquement les clés
                    annotated_texts.append((text, {'glirel_labels': labels}))
            docs = self.nlp.pipe(annotated_texts, as_tuples=True)
            results = []
            for doc in docs:
                if not doc.ents:
                    logger.warning("No entities found in document.")
                    results.append([])
                    continue
                filtered_ents = filter_spans(doc.ents)
                results.append([
                    {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
                    for ent in filtered_ents
                ])
            return results
        except Exception as e:
            logger.error(f"Error during entity extraction: {e}")
            return []


    def annotate_relations(self, text: str, labels: Dict) -> List[Dict]:
        doc = self.nlp(text)
        relations = [
            {
                "head_text": rel["head_text"],
                "tail_text": rel["tail_text"],
                "label": rel["label"],
                "score": rel["score"],
            }
            for rel in doc._.relations
        ]
        logger.info(f"Extracted relations: {relations}")
        return relations

    def extract_relations(self, texts: List[str], labels: Dict) -> List[List[Dict]]:
        return [self.annotate_relations(text, labels) for text in texts]

    def annotate_sentiments(self, text: str) -> List[Dict]:
        sentiment_model = spacy.load("fr_core_news_md")
        doc = sentiment_model(text)
        sentiments = [
            {
                "text": sent.text,
                "sentiment": sent._.polarity if hasattr(sent._, "polarity") else "neutral",
            }
            for sent in doc.sents
        ]
        logger.info(f"Extracted sentiments: {sentiments}")
        return sentiments

    def annotate_coreferences(self, text: str) -> List[Dict]:
        coref_model = spacy.load("en_coref_md")  # Replace with actual coreference model
        doc = coref_model(text)
        resolved_text = doc._.coref_resolved
        logger.info(f"Resolved coreferences for text '{text[:100]}': {resolved_text}")
        return [{"text": text, "resolved": resolved_text}]

    def annotate_events(self, text: str) -> List[Dict]:
        events = []
        doc = self.nlp(text)
        for sent in doc.sents:
            for token in sent:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    event = {
                        "event": token.lemma_,
                        "subject": [child.text for child in token.children if child.dep_ in ("nsubj", "nsubjpass")],
                        "object": [child.text for child in token.children if child.dep_ in ("dobj", "attr", "prep")],
                    }
                    events.append(event)
        logger.info(f"Extracted events: {events}")
        return events
