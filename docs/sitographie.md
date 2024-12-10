# Bibliographie MLOps, NLP, GLiNER, GLiREL, LangChain, Docling v2
---

## Docling v2

- **Docling v2 : Préparez vos documents pour l'IA générative**  
  [Docling v2 Documentation](https://ds4sd.github.io/docling/v2/)  
  Présentation des fonctionnalités et améliorations de Docling v2, outil de préparation de documents pour l'IA générative.

- **Dépôt GitHub de Docling v2**  
  [GitHub - dsrdatta/docling_v2](https://github.com/dsrdatta/docling_v2)  
  Code source et documentation de Docling v2.

 [Docling: A Versatile Document Parsing Tool - Medium](https://medium.com/ai-artistry/docling-a-versatile-document-parsing-tool-8db098fcfb2e)

[Building Document Parsing Pipelines with Python - Medium](https://lasha-dolenjashvili.medium.com/building-document-parsing-pipelines-with-python-3c06f62569ad)
---

## GLiNER et GLiREL (NER et Extraction de Relations)

### GLiNER : Generalist and Lightweight Model for Named Entity Recognition
- **Article GLiNER :**  
  *Urchade Zaratiana, Nadi Tomeh, Pierre Holat, Thierry Charnois*  
  [arXiv:2311.08526](https://arxiv.org/abs/2311.08526)  
  Présentation d’un modèle généraliste et léger pour la reconnaissance d'entités nommées.

- **Dépôt GitHub de GLiNER**  
  [GitHub - urchade/GLiNER](https://github.com/urchade/GLiNER)  
  Code source, modèles pré-entraînés et exemples d'utilisation.

- [🤗 Hugging Space by Tom Aarsen](https://huggingface.co/spaces/tomaarsen/gliner_medium-v2.1)

- **Google Colab - "GLiNER-Studio" by Knowledgator** 
  [Gliner_Studio.ipynb](https://colab.research.google.com/github/Knowledgator/GLiNER-Studio/blob/main/notebooks/Gliner_Studio.ipynb)

### GLiREL : Generalist and Lightweight Model for Relation Extraction
- **Dépôt GitHub de GLiREL**  
  [GitHub - jackboyla/GLiREL](https://github.com/jackboyla/GLiREL)  
  Code source, modèles pré-entraînés et exemples d'utilisation d'un modèle généraliste et léger pour l'extraction de relations.

- **"GLiNER : le succès du modèle de reconnaissance d'entités nommées par F.initiatives"** 
[Le modèle GLiNER par F.initiatives](https://www.f-initiatives.com/actualites/rd/gliner-le-succes-du-modele-de-reconnaissance-dentites-nommees-par-f-initiatives/)


### Intégration GLiNER / GLiREL avec LangChain
- **GlinerGraphTransformer**  
  [LangChain Documentation](https://api.python.langchain.com/en/latest/graph_transformers/langchain_experimental.graph_transformers.gliner.GlinerGraphTransformer.html)  
  Permet de convertir des documents en graphes via GLiNER/GLiREL.

- **GLiNERLinkExtractor**  
  [LangChain Documentation](https://api.python.langchain.com/en/latest/community/graph_vectorstores/langchain_community.graph_vectorstores.extractors.gliner_link_extractor.GLiNERLinkExtractor.html)  
  Extracteur de liens entre documents partageant des entités nommées.


[Deep Dive "sous le capot" du modèle - Medium](https://medium.com/@zilliz_learn/gliner-generalist-model-for-named-entity-recognition-using-bidirectional-transformer-ed65165a4877)
---

[Enhancing Retrieval-Augmented Generation: Tackling Polysemy, Homonyms and Entity Ambiguity with GLiNER for Improved Performance](https://medium.com/@mollelmike/enhancing-retrieval-augmented-generation-tackling-polysemy-homonyms-and-entity-ambiguity-with-0fa4d395c863)

[GLiNER: A Zero-Shot NER that outperforms ChatGPT and traditional NER models](https://netraneupane.medium.com/gliner-zero-shot-ner-outperforming-chatgpt-and-traditional-ner-models-1f4aae0f9eef)

[Understanting PII Anonymization](https://medium.com/@jubelahmed/understanding-pii-anonymization-with-python-a-simple-guide-68863cc0d129)

[Semantic Chunking for RAG - Medium](https://medium.com/the-ai-forum/semantic-chunking-for-rag-f4733025d5f5)

[Introducing GraphRAG with LangChain and Neo4j](https://medium.com/microsoftazure/introducing-graphrag-with-langchain-and-neo4j-90446df17c1e)


[Setup collaborative MLflow with PostgreSQL as Tracking Server and MinIO as Artifact Store using docker containers](https://medium.com/@amir_masoud/setup-collaborative-mlflow-with-postgresql-as-tracking-server-and-minio-as-artifact-store-using-45c76a9d9814)

[How to Run Llama 3.2-Vision Locally With Ollama: A Game Changer for Edge AI](https://medium.com/@tapanbabbar/how-to-run-llama-3-2-vision-on-ollama-a-game-changer-for-edge-ai-80cb0e8d8928)
---

[MLflow Tracking Server](https://mlflow.org/docs/2.17.2/tracking/server.html)
[MLflow (2.17.2) Model Registry](https://mlflow.org/docs/2.17.2/model-registry.html#adding-an-mlflow-model-to-the-model-registry)
## Docker, FastAPI, et Configurations Réseau

18. [Setting Up FastAPI](https://fastapi.tiangolo.com/deployment/docker/)  
20. [Docker Networking Simplified](https://docs.docker.com/network/)  
21. [MinIO with Traefik](https://min.io/docs/minio/linux/)  
22. [Securing FastAPI with OAuth2](https://auth0.com/)  
24. [Traefik Documentation](https://doc.traefik.io/traefik/)  
25. [PostgreSQL in Docker Containers](https://hub.docker.com/_/postgres)   
27. [Cloudflared](https://github.com/cloudflare/cloudflared)

---

## Embeddings, LangChain, Ollama et Outils ML

28. [Generating Embeddings with Ollama](https://ollama.ai/)  
29. [Using Hugging Face Models for Embedding Tasks](https://huggingface.co/docs/)  

### LangChain
36. [LangChain Documentation](https://python.langchain.com/)  
37. [LangChain API Reference](https://api.python.langchain.com/)  
38. [Chaining LLMs with LangChain](https://blog.langchain.com/)  
39. [LangChain for Large Language Models (Medium)](https://medium.com/)  
40. [Wikipedia - LangChain](https://en.wikipedia.org/wiki/LangChain)

---

## Label Studio et Traitement de Documents

41. [Label Studio Official Documentation](https://labelstud.io/)  
42. [Building Annotation Interfaces with Label Studio](https://labelstud.io/blog/)  
43. [Integrating Label Studio with FastAPI (GitHub)](https://github.com/)  
44. [PostgreSQL and Label Studio Integration](https://postgresql.org/)  
45. [Using Label Studio for Document Processing](https://konfuzio.com/)  
46. [JSON, YAML, and Markdown in Document Workflows](https://json-schema.org/)  
47. [Multi-Format Document Handling with Python](https://docs.python.org/)


---