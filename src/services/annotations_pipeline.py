# src/services/annotation_pipelines.py
from transformers import AutoModelForTokenClassification, pipeline
from glirel import GLiREL
import glirel
from gliner import GLiNER
from gliner_spacy.pipeline import GlinerSpacy
from src.config import Settings
from typing import List, Dict
import spacy
from spacy.util import registry
from spacy.language import Language
from loguru import logger

@Language.factory("gliner_spacy_factory")
def create_gliner_spacy(nlp: Language, name: str, gliner_model: str, device: str) -> GlinerSpacy:
    """Factory function for GLiNER component."""
    return GlinerSpacy(
        nlp=nlp,
        name=name,
        gliner_model="urchade/gliner_smallv2.1",
        chunk_size=250,
        labels=["person", "organization", "email"],
        style="ent",
        threshold=0.3,
        map_location=device,
    )


@Language.factory("glirel_factory")
def create_glirel(nlp: Language, name: str, glirel_model: str, device: str) -> GLiREL:
    """Factory function for GLiREL component."""
    return GLiREL.from_pretrained(
        pretrained_model_name_or_path=glirel_model,
        device=device,
         use_fast=False
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
            config={"gliner_model": config["gliner_model"], "device": config["device"]}
        )

        # Ajouter GLiREL
        nlp.add_pipe(
            "glirel_factory", 
            name="glirel", 
            after="gliner_spacy", 
            config={"glirel_model": config["glirel_model"], "device": config["device"]}
        )

        return nlp

    def annotate_ner(self, text: str) -> List[Dict]:
        """Extract Named Entities using GLiNER."""
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            try:
                start = int(ent.start_char)
                end = int(ent.end_char)
                if start >= end:
                    logger.warning(f"Entity with invalid range: {ent.text} ({start}-{end})")
                    continue
                entity = {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": start,
                    "end": end,
                }
                entities.append(entity)
            except ValueError as ve:
                logger.error(f"Invalid entity range: {ve} in text '{text}'")
            except Exception as e:
                logger.error(f"Unexpected error processing entity '{ent.text}': {e}")
        logger.info(f"Extracted entities: {entities}")
        return entities

    def extract_entities(self, texts: List[str]) -> List[List[Dict]]:
        """Batch extraction of Named Entities."""
        results = []
        for text in texts:
            entities = self.annotate_ner(text)
            # Validation stricte des entités extraites
            for entity in entities:
                if not isinstance(entity["start"], int) or not isinstance(entity["end"], int):
                    logger.error(f"Entity has invalid types: {entity}")
                    raise ValueError(f"Invalid entity detected: {entity}")
            results.append(entities)
        logger.info(f"Batch entity extraction results: {results}")
        return results

    def annotate_relations(self, text: str, labels: Dict) -> List[Dict]:
        """Extract relations using GLiREL."""
        doc = self.nlp(text)
        relations = [
            {
                "subject": rel["subject"],
                "relation": rel["relation"],
                "object": rel["object"],
                "confidence": rel["confidence"],
            }
            for rel in doc._.relations
        ]
        logger.info(f"Extracted relations: {relations}")
        return relations

    def extract_relations(self, texts: List[str], labels: Dict) -> List[List[Dict]]:
        """Batch extraction of relations."""
        results = [self.annotate_relations(text, labels) for text in texts]
        logger.info(f"Batch relation extraction results: {results}")
        return results

    def annotate_sentiments(self, text: str) -> List[Dict]:
        """Sentiment Analysis using spaCy."""
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
        """Coreference resolution using spaCy or an external pipeline."""
        # Placeholder for coreference resolution integration
        coref_model = spacy.load("en_coref_md")  # Replace with actual coreference model
        doc = coref_model(text)
        resolved_text = doc._.coref_resolved
        logger.info(f"Resolved coreferences for text '{text[:100]}': {resolved_text}")
        return [{"text": text, "resolved": resolved_text}]

    def annotate_events(self, text: str) -> List[Dict]:
        """Extract events from the text."""
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