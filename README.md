#### 📜 Pour plus de détails, consultez la documentation complète sur la [Documentation & Guide du Projet](https://artemis-ia.github.io/weavestruct/).

---

# 🌀 **WeaveStruct**  
> **De la donnée brute à l'information exploitable : traitement de documents, extraction d'entités et relations, et construction de graphes de connaissances.**  

WeaveStruct est une plateforme modulaire et évolutive dédiée au traitement intelligent de documents. Grâce à l'intégration de technologies avancées en NLP et en Machine Learning, elle permet d'analyser, structurer et exploiter des données non structurées en informations prêtes à l'emploi.

---

## 🚀 **Fonctionnalités clés**
### 📄 **Traitement des documents**
- **Formats pris en charge** : PDF, DOCX, avec prise en charge de l'OCR pour les documents scannés.
- **Extraction avancée** : Extraction des tableaux, figures, et métadonnées.
- **Conversion multi-format** : Export en JSON, YAML, Markdown ou CSV pour une intégration aisée.

### 🧠 **Reconnaissance d'entités et extraction de relations**
- **Modèles avancés** :  
  - **GLiNER** : Reconnaissance d'entités nommées (personnes, organisations, lieux, etc.).
  - **GLIREL** : Extraction de relations logiques et hiérarchiques.
- **Résultats exploitables** : Stockage des entités et relations dans une base de données orientée graphes (Neo4j).

### 🌐 **Construction de graphes de connaissances**
- **Stockage relationnel** : Construction automatique de graphes dans Neo4j.
- **Visualisation intuitive** : Données prêtes pour des outils de visualisation tels que Cytoscape ou GraphXR.

### 🔍 **Recherche vectorielle et embeddings**
- **Modèles d'embeddings** : Intégration du modèle Ollama pour transformer les documents en représentations vectorielles.
- **Indexation rapide** : Recherche vectorielle rapide grâce à PostgreSQL et l'extension PGVector.

### 📈 **Suivi des performances et monitoring**
- **Tracking des modèles** : Intégration avec MLflow pour le suivi des expériences et métriques.
- **Monitoring système** : Metrics exposées pour Prometheus pour une supervision en temps réel.

---

## 🧱 **Briques utilisées**
- **[DoclingV2](https://github.com/your-doclingv2-link)** : Framework avancé pour le traitement et l'analyse de documents. C'est le point d'entrée des données.
- **[LangChain](https://github.com/hwchase17/langchain)** : Gestion des flux conversationnels et chaînes d'appels de modèles pour des cas complexes, avec des classes clés comme :  
  - **[`LinkExtractor`](https://python.langchain.com/api_reference/community/graph_vectorstores/langchain_community.graph_vectorstores.extractors.gliner_link_extractor.GLiNERLinkExtractor.html)** : Extraction des liens logiques entre les entités mentionnées dans un document.  
  - **[`GraphTransformer`](https://python.langchain.com/api_reference/experimental/graph_transformers/langchain_experimental.graph_transformers.gliner.GlinerGraphTransformer.html#glinergraphtransformer)** : Transformation des données textuelles en graphes exploitables.  
- **[GLiNER](https://github.com/urchade/GLiNER)** : Reconnaissance d'entités nommées à l'aide de modèles NLP préentraînés.  
- **[Ollama](https://www.ollama.ai/)** : Génération d'embeddings vectoriels et analyse de documents pour la recherche vectorielle.  

---

## ⚙️ **API : Points d'entrée principaux et description**
### 📂 **Gestion des documents**
- `POST /documents/upload/`  
  **Description** : Télécharge un document pour traitement initial (extraction de texte, OCR, etc.).
  
- `POST /documents/index_document/`  
  **Description** : Indexe un document pour exécuter des tâches d'extraction d'entités et de relations.

- `POST /documents/rag_process/`  
  **Description** : Convertit un document en embeddings vectoriels pour une recherche rapide.

### 🔗 **Graphes de connaissances**
- `GET /graph/entities/`  
  **Description** : Renvoie toutes les entités extraites et enregistrées dans la base de données de graphes.

- `GET /graph/relationships/`  
  **Description** : Récupère toutes les relations entre les entités identifiées.

- `GET /graph/visualize/`  
  **Description** : Renvoie les données formatées pour visualiser le graphe des entités et relations.

### 🔍 **Recherche**
- `GET /search/entities/`  
  **Description** : Permet de rechercher des entités spécifiques dans la base à l'aide de mots-clés.

- `GET /search/relationships/`  
  **Description** : Effectue une recherche sur les relations existantes dans le graphe.

### 🛠️ **Administration et suivi**
- `GET /metrics/`  
  **Description** : Expose les métriques système pour le monitoring via Prometheus.

---

## 🌟 **Contribuer**
Les contributions sont les bienvenues !  
1. Forkez le projet.  
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/awesome-feature`).  
3. Commitez vos modifications (`git commit -m 'Add awesome feature'`).  
4. Poussez la branche (`git push origin feature/awesome-feature`).  
5. Ouvrez une Pull Request.

---

## 📜 **Licence**
Ce projet est sous licence MIT. Consultez le fichier [LICENSE](LICENSE) pour plus d'informations.

---

## 📞 **Support**
- **Issues** : N'hésitez pas à signaler des problèmes via la section [Issues](https://github.com/Artemis-IA/weavestruct/issues).  
---

Pour plus de détails, consultez la documentation complète sur notre [GitHub Page](https://github.com/Artemis-IA/weavestruct).

---
