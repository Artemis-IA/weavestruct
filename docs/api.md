## **API : Routes et Usages**

L'API de WeaveStruct fournit divers endpoints pour la gestion de documents, la construction de graphes, et generation d'embeddings:

---

### **1. Gestion des Documents**

#### **POST `/documents/upload/`**
- **Description** : Télécharge et traite des fichiers pour l'extraction de texte, figures, tableaux, et métadonnées.
- **Paramètres** :
  - **`files`** : Liste des fichiers à uploader (requis).
  - **`export_formats`** : Liste des formats d'exportation, par exemple `JSON`, `YAML`, `Markdown` (par défaut : `JSON`).
  - **`use_ocr`** : Activer l'OCR pour les documents scannés (par défaut : `false`).
  - **`export_figures`** : Exporter les figures (par défaut : `true`).
  - **`export_tables`** : Exporter les tableaux (par défaut : `true`).
  - **`enrich_figures`** : Ajouter des métadonnées aux figures (par défaut : `false`).
- **Réponse** : 
  - `success_count` : Nombre de fichiers traités avec succès.
  - `failure_count` : Nombre de fichiers échoués.

#### **POST `/documents/index_document/`**
- **Description** : Indexe un document en extrayant les entités et en générant des embeddings.
- **Paramètres** :
  - **`file`** : Document à indexer (optionnel).
  - **`export_formats`** : Liste des formats d'exportation (par défaut : `JSON`).
- **Réponse** : 
  - Message confirmant l'indexation.
  - Statut des entités extraites.

---

### **2. Graphes de Connaissances**

#### **GET `/graph/entities/`**
- **Description** : Récupère toutes les entités extraites.
- **Paramètres** : Aucun.
- **Réponse** : Liste des entités, incluant leurs attributs et métadonnées.

#### **GET `/graph/relationships/`**
- **Description** : Récupère toutes les relations entre les entités.
- **Paramètres** : Aucun.
- **Réponse** : Liste des relations avec leurs détails.

#### **GET `/graph/visualize/`**
- **Description** : Génère des données formatées pour visualiser le graphe des entités et relations.
- **Paramètres** : Aucun.
- **Réponse** : Données prêtes pour des outils de visualisation.

---

### **3. Recherche**

#### **GET `/search/entities/`**
- **Description** : Recherche des entités par mot-clé.
- **Paramètres** :
  - **`query`** : Mot-clé ou expression à rechercher (requis).
- **Réponse** : Liste des entités correspondantes.

#### **GET `/search/relationships/`**
- **Description** : Recherche des relations par mot-clé.
- **Paramètres** :
  - **`query`** : Mot-clé ou expression à rechercher (requis).
- **Réponse** : Liste des relations correspondantes.

---

### **4. Relations**

#### **POST `/relationships/`**
- **Description** : Crée une relation entre deux entités.
- **Paramètres** :
  - **`type`** : Type de la relation (requis).
  - **`source_id`** : ID de l'entité source (requis).
  - **`target_id`** : ID de l'entité cible (requis).
- **Réponse** : Objet de la relation créée.

#### **GET `/relationships/`**
- **Description** : Liste toutes les relations.
- **Paramètres** : Aucun.
- **Réponse** : Liste des relations.

#### **DELETE `/relationships/{relationship_id}`**
- **Description** : Supprime une relation.
- **Paramètres** :
  - **`relationship_id`** : Identifiant de la relation (requis).
- **Réponse** : Message confirmant la suppression.

---

### **5. Administration**

#### **POST `/log_models/`**
- **Description** : Enregistre des informations sur les modèles dans les logs.
- **Paramètres** : Aucun.
- **Réponse** : Succès ou échec.

#### **POST `/log_queries/`**
- **Description** : Enregistre des informations sur les requêtes effectuées.
- **Paramètres** :
  - **`query`** : Requête à enregistrer (requis).
- **Réponse** : Succès ou échec.

#### **GET `/metrics/`**
- **Description** : Expose des métriques pour le monitoring.
- **Paramètres** : Aucun.
- **Réponse** : Données de monitoring.

---

### **6. Ensembles de Données**

#### **POST `/datasets/`**
- **Description** : Crée un ensemble de données à partir de fichiers téléchargés.
- **Paramètres** :
  - **`name`** : Nom de l'ensemble de données (optionnel).
  - **`files`** : Fichiers à inclure dans l'ensemble de données (requis).
  - **`labels`** : Étiquettes associées (optionnel).
  - **`output_format`** : Format de sortie (par défaut : `json-ner`).
- **Réponse** : Ensemble de données créé.

#### **GET `/datasets/`**
- **Description** : Liste tous les ensembles de données.
- **Paramètres** : Aucun.
- **Réponse** : Liste des ensembles de données.

#### **DELETE `/datasets/{dataset_id}`**
- **Description** : Supprime un ensemble de données.
- **Paramètres** :
  - **`dataset_id`** : ID de l'ensemble de données à supprimer.
- **Réponse** : Message confirmant la suppression.

---

## **Exemple d'utilisation**

### **Créer une entité**
**Requête** : 
```json
POST /entities/
{
  "name": "Company X",
  "type": "Organization"
}
```

**Réponse** : 
```json
{
  "id": "12345",
  "name": "Company X",
  "type": "Organization"
}
```

---

### **Rechercher une entité**
**Requête** :
```json
GET /search/entities/?query=Company
```

**Réponse** : 
```json
[
  {
    "id": "12345",
    "name": "Company X",
    "type": "Organization"
  }
]
