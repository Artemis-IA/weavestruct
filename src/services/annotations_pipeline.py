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

@Language.factory("gliner_spacy_factory")
def create_gliner_spacy(nlp: Language, name: str, gliner_model: str, device: str) -> GlinerSpacy:
    """Factory function for GLiNER component."""
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
        """Annotate Named Entities using GLiNER."""
        doc = self.nlp(text)
        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "score": ent._.score if hasattr(ent._, "score") else None,
            }
            for ent in doc.ents
        ]
        return entities

    def annotate_relations(self, text: str, labels: Dict) -> List[Dict]:
        """
        Annotate Relations using GLiREL.
        :param text: Input text
        :param labels: Relation extraction labels (as required by GLiREL)
        """
        docs = list(self.nlp.pipe([(text, labels)], as_tuples=True))
        relations = docs[0][0]._.relations
        return [
            {
                "head_text": rel["head_text"],
                "label": rel["label"],
                "tail_text": rel["tail_text"],
                "score": rel["score"],
            }
            for rel in relations
        ]

    def annotate_sentiments(self, text: str) -> List[Dict]:
        """Sentiment Analysis using spaCy."""
        # Load the sentiment model
        sentiment_model = spacy.load("fr_core_news_md")
        doc = sentiment_model(text)
        sentiments = [
            {
                "text": sent.text,
                "sentiment": sent._.polarity if hasattr(sent._, "polarity") else "neutral",
            }
            for sent in doc.sents
        ]
        return sentiments

    def annotate_coreferences(self, text: str) -> List[Dict]:
        """Coreference resolution using spaCy or an external pipeline."""
        # Placeholder for coreference resolution integration
        # Example using a hypothetical coreference model
        coref_model = spacy.load("en_coref_md")  # Replace with actual coreference model
        doc = coref_model(text)
        resolved_text = doc._.coref_resolved
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
        return events
