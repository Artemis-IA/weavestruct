### Diagramme Séquentiel Formaté en Markdown

Voici une version bien structurée et lisible, adaptée à un diagramme séquentiel avec **inputs/outputs détaillés** et **toutes les étapes clés**.

```plaintext
Client                 API WeaveStruct                MinIO                    PostgreSQL                Neo4j                  Ollama/GLiNER             MLOps Backend             Loop ML
   |                         |                          |                          |                         |                          |                          |                          |
   |-- POST /documents/upload -->|                      |                          |                         |                          |                          |                          |
   |                          |-- Valide et sauvegarde fichiers : ["doc1.pdf", "doc2.docx"] -->|              |                          |                         |                          |                          |
   |                          |                          |<-- Confirmation de sauvegarde : ["doc1.pdf", "doc2.docx"]--|                  |                         |                          |                          |
   |                          |-- Enregistre métadonnées (export_formats: ["json", "yaml"], use_ocr: true, enrich_figures: true) -->|--> PostgreSQL            |                          |                          |                          |
   |                          |                          |                          |                         |                          |                          |                          |
   |<-- Réponse succès : { status: "success", results: [{ "file": "doc1.json", "size": "15KB" }, { "file": "doc2.yaml", "size": "20KB" }] } --------|                  |                          |                          |                          |
   |                          |                          |                          |                         |                          |                          |                          |
   |-- POST /documents/index_documents -->|             |                          |                         |                          |                          |                          |
   |                          |-- Récupère fichier "doc1.pdf" -->|                |                         |                          |                          |                          |
   |                          |                          |<-- Retourne fichier : "doc1.pdf"--|                |                          |                          |                          |
   |                          |-- Génère embeddings pour "doc1" -->|------------->|                         |                          |                          |                          |
   |                          |                          |                          |                         |                          |                          |                          |
   |                          |-- Extraction des entités (GLiNER) et relations (GLiREL) -->|                 |                         |                          |                          |
   |                          |                          |                          |<-- Entités : ["nom", "date"]                       |                          |                          |                          |
   |                          |                          |                          |<-- Relations : ["travaille_pour", "situe_a"]       |                          |                          |                          |
   |                          |-- Sauvegarde entités et relations -->|------------------------->|           |                          |                          |                          |
   |                          |                          |                          |                         |                          |                          |                          |
   |<-- Réponse succès : { status: "indexed", knowledge_graph: ["graph_link"] } --------|                     |                          |                          |                          |
   |                          |                          |                          |                         |                          |                          |                          |
   |-- POST /loopml/train -->|                          |                          |                         |                          |                          |                          |
   |                          |-- Envoie données entraînement (fichiers : ["dataset.json"], labels) -->|     |                          |                          |                          |
   |                          |                          |                          |                         |<-- Modèle GLiNER/REL entraîné : "model_v1.0"--|         |
   |                          |                          |                          |                         |-- Sauvegarde artefacts MLFlow -->|                          |
   |<-- Réponse succès : { status: "trained", model_version: "1.0" } --------|                          |                          |                         |                          |                          |
   |                          |                          |                          |                         |                          |                          |                          |
   |-- GET /loopml/predict -->|                          |                          |                         |                          |                          |                          |
   |                          |-- Charge modèle : "model_v1.0" -->|              |                         |                          |                          |                          |
   |                          |                          |                          |                         |-- Prédictions (NER/REL) pour "doc3.pdf" -->|              |
   |                          |                          |                          |                         |<-- Résultat : ["entités", "relations"]--|              |
   |                          |-- Sauvegarde prédictions -->|------------------------->|                     |                          |                          |                          |
   |<-- Réponse succès : { predictions: ["entités", "relations"] } --------|                          |                          |                         |                          |                          |
   |                          |                          |                          |                         |                          |                          |                          |
   |-- GET /knowledge-graph -->|                          |                          |                         |                          |                          |                          |
   |                          |                          |                          |------------------------->|                          |                          |                          |
   |                          |                          |                          |<-- Retourne graphe connaissances : "graph_url"--|                          |                          |
   |<-- Graphe de connaissances : { link: "graph_url" } -------------------|                          |                          |                         |                          |                          |
```

### Points Clés :

1. **Chaque interaction listée avec détails d'entrée et sortie** :
   - Exemple : 
     - `POST /documents/upload` inclut `files`, `export_formats`, et toutes les métadonnées nécessaires.
     - La réponse inclut les liens de fichiers exportés.

2. **Interaction entre services clairement définie** :
   - Communication entre **API**, **MinIO**, **PostgreSQL**, **Neo4j**, et modèles (**GLiNER**, **MLFlow**, etc.).

3. **Réponses détaillées** :
   - Chaque réponse contient des informations précises (statut, liens vers les résultats, données générées).

4. **Facilement intégrable dans un diagramme séquentiel** :
   - Copiez ce texte pour l'utiliser dans un outil comme **Draw.io** ou un éditeur Markdown prenant en charge la prévisualisation de diagrammes.