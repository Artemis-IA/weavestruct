# Guide d'utilisation

## Documents

### Upload
- **Endpoint** : `/documents/upload`
- **Description** : Permet de télécharger et traiter des fichiers.
- **Méthode** : `POST`
- **Paramètres** :
  - `files`: Liste des fichiers
  - `export_formats`: Formats d'exportation (JSON, YAML, Markdown)

```json
{
  "message": "File processed successfully",
  "success_count": 1,
  "failure_count": 0
}
