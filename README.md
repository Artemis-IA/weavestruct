### README.md

# **WeaveStruct : Système Avancé de Traitement de Documents et Construction de Graphes de Connaissances**

**WeaveStruct** est une plateforme avancée pour le **traitement de documents**, la **reconnaissance d'entités nommées (NER)**, l'extraction de **relations**, et la construction de **graphes de connaissances**. Ce projet repose sur un ensemble de technologies modernes pour transformer des données non structurées en informations exploitables, tout en offrant des capacités de suivi et de monitoring.

---

## **Caractéristiques**

### **Traitement de Documents**
- Prise en charge des formats **PDF** et **DOCX** avec options pour l’OCR.
- Extraction des **tableaux**, **figures**, et leur export en **JSON**, **YAML**, **Markdown** ou **CSV**.
- Nettoyage et prétraitement du texte (normalisation et suppression des caractères indésirables).

### **Reconnaissance d'Entités et Extraction de Relations**
- **NER (Named Entity Recognition)** : Identification des entités comme les personnes, organisations, lieux, etc., grâce à des modèles avancés (`GLiNER`).
- **Relation Extraction** : Détection des liens logiques entre entités avec des modèles tels que `GLIREL`.

### **Construction de Graphes de Connaissances**
- **Stockage dans Neo4j** : Représentation structurée des entités et relations dans une base orientée graphes.
- **Visualisation des relations** : Génération de données prêtes pour des outils de visualisation.

### **Recherche Vectorielle et Embeddings**
- Génération d’**embeddings vectoriels** pour les documents avec le modèle `Ollama`.
- Recherche vectorielle rapide dans **PostgreSQL** avec l’extension **PGVector**.

### **Pipeline Modulaire et Suivi**
- **Extensible** : Architecture permettant l'ajout ou la modification de modules pour des cas d'usage spécifiques.
- **Suivi des modèles et métriques** avec **MLflow**.
- **Monitoring des performances** via **Prometheus**.

---

## **Technologies Utilisées**

### Backend
- **[FastAPI](https://fastapi.tiangolo.com/)** : Framework rapide et moderne pour construire des APIs avec Python.
- **[SQLAlchemy](https://www.sqlalchemy.org/)** : ORM pour interagir avec PostgreSQL.

### Bases de Données et Stockage
- **[PostgreSQL](https://www.postgresql.org/)** : Base de données relationnelle robuste.
- **[PGVector](https://github.com/pgvector/pgvector)** : Extension pour la recherche vectorielle dans PostgreSQL.
- **[Neo4j](https://neo4j.com/)** : Base de données orientée graphes pour la gestion des relations entre entités.
- **[MinIO](https://min.io/)** : Système de stockage compatible S3 pour les fichiers et résultats analytiques.

### Machine Learning
- **[GLiNER](https://github.com/E3-JSI/gliner)** : Modèle avancé pour la reconnaissance d’entités nommées.
- **[GLIREL](https://huggingface.co/models)** : Modèle pour l’extraction des relations entre entités.
- **[Ollama](https://ollama.ai/)** : Génération d’embeddings pour documents et requêtes.
- **[Transformers](https://huggingface.co/docs/transformers/)** : Bibliothèque pour utiliser des modèles pré-entraînés.

### Monitoring et Gestion
- **[Prometheus](https://prometheus.io/)** : Solution open-source pour le monitoring et la collecte de métriques.
- **[MLflow](https://mlflow.org/)** : Plateforme pour le suivi des expériences et des modèles.
- **[Loguru](https://github.com/Delgan/loguru)** : Bibliothèque de gestion de logs simple et puissante.

---

## **Installation et Configuration**

### Prérequis

1. Installez Python 3.8 ou supérieur.
2. Configurez Docker pour les services (PostgreSQL, Neo4j, MinIO).

### Installation des Dépendances

1. Clonez le dépôt :
   ```bash
   git clone <URL_DU_DÉPÔT>
   cd weavestruct
   ```

2. Installez les dépendances Python :
   ```bash
   pip install -r requirements.txt
   ```

3. Configurez les variables d’environnement dans un fichier `.env` :
   ```env
   DATABASE_URL=postgresql://user:password@localhost:5432/weavestruct
   NEO4J_URI=bolt://neo4j:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   MINIO_URL=http://minio:9000
   ```

4. Lancez les services nécessaires :
   ```bash
   docker-compose up -d
   ```

### Démarrage de l’Application

1. Exécutez le serveur FastAPI :
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. Accédez à l’interface interactive de l’API : [http://localhost:8000](http://localhost:8000).

---

## **API : Points d'Entrée Principaux**

### Documents
- **`POST /documents/upload/`** : Téléchargez et analysez un ou plusieurs documents.
- **`POST /documents/index_document/`** : Indexe un document pour extraire les entités et relations.
- **`POST /documents/rag_process/`** : Traite un document pour le transformer en embeddings vectoriels.

### Graphes de Connaissances
- **`GET /graph/entities/`** : Récupérez toutes les entités enregistrées dans le graphe.
- **`GET /graph/relationships/`** : Récupérez toutes les relations extraites.
- **`GET /graph/visualize/`** : Obtenez des données de visualisation des relations.

### Jeux de Données
- **`POST /datasets/`** : Créez un jeu de données annoté à partir de fichiers téléchargés.
- **`POST /train/`** : Entraînez un modèle de reconnaissance d’entités sur un dataset spécifique.

### Recherche
- **`GET /search/entities/`** : Recherchez des entités à l’aide de mots-clés.
- **`GET /search/relationships/`** : Recherchez des relations spécifiques.

### Monitoring
- **`GET /metrics/`** : Expose les métriques Prometheus pour le suivi de la performance.

---

## **Exemple d'Utilisation**

### Étapes pour Analyser un Document PDF

1. **Téléchargez le fichier** via l’API :
   ```bash
   curl -X POST "http://localhost:8000/documents/upload/" -F "files=@document.pdf"
   ```

2. **Récupérez les entités extraites** :
   ```bash
   curl -X GET "http://localhost:8000/graph/entities/"
   ```

3. **Visualisez les relations** dans un graphe :
   ```bash
   curl -X GET "http://localhost:8000/graph/visualize/"
   ```

---

## **Surveillance et Suivi**

- **Prometheus** : Collecte des métriques sur les performances et ressources système (exposées à `/metrics`).
- **Logs** : Gestion centralisée des logs avec Loguru.

---

## **Contributeur**

- **P3 Simplon**  
  Développeur principal

Pour toute suggestion ou contribution, merci de créer une **issue** ou une **pull request** sur le dépôt GitHub.