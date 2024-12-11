import os
import streamlit as st
from pathlib import Path
import boto3
import json
import yaml

def get_s3_client():
    """Crée un client S3 en utilisant les variables d'environnement."""
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_URL", "http://localhost:9000"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minio"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minio123"),
    )

def fetch_file_from_s3(s3_client, bucket_name, file_key):
    """
    Récupère un fichier depuis S3 et retourne son contenu binaire.
    """
    obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    return obj["Body"].read()

def main():
    st.set_page_config(layout="wide", page_title="Document Viewer")

    st.title("Visualisation du Document et de ses Conversions")

    # Configuration du bucket S3
    s3_client = get_s3_client()
    input_bucket = os.getenv("INPUT_BUCKET", "docs-input")

    # Barre latérale pour sélectionner un document
    st.sidebar.title("Sélection du Document")
    response = s3_client.list_objects_v2(Bucket=input_bucket, Prefix="input/")
    files = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].lower().endswith(".pdf")]

    if not files:
        st.sidebar.warning("Aucun fichier PDF disponible dans le bucket S3.")
        return

    selected_file = st.sidebar.selectbox("Choisissez un document PDF", files)

    # Mise en page : deux colonnes
    col1, col2 = st.columns([2, 3])

    # Colonne de gauche : document original
    with col1:
        st.header("Document Original (PDF)")
        try:
            pdf_bytes = fetch_file_from_s3(s3_client, input_bucket, selected_file)
            # Bouton de téléchargement pour le PDF original
            st.download_button(
                label="Télécharger le PDF original",
                data=pdf_bytes,
                file_name=Path(selected_file).name,
                mime="application/pdf",
            )
            # Affichage du PDF dans Streamlit
            st.pdf(pdf_bytes)
        except Exception as e:
            st.error(f"Erreur lors de la récupération du PDF: {e}")

    # Colonne de droite : formats convertis
    with col2:
        st.header("Formats Convertis")

        # Formats disponibles
        output_formats = ["yaml", "json", "md", "txt"]
        format_selector = st.selectbox("Sélectionnez un format de sortie", output_formats)

        # On suppose que les documents convertis sont stockés dans des buckets séparés ou chemins préfixés
        # Exemple : "docs-output-json", "docs-output-yaml", "docs-output-md", "docs-output-txt"
        # Ajustez selon votre architecture.
        bucket_map = {
            "json": "docs-output-json",
            "yaml": "docs-output-yaml",
            "md": "docs-output-md",
            "txt": "docs-output-txt"
        }

        output_bucket = bucket_map[format_selector]
        output_file_key = Path(selected_file).stem + f".{format_selector}"

        try:
            file_content = fetch_file_from_s3(s3_client, output_bucket, str(output_file_key))

            if format_selector == "json":
                content = json.loads(file_content)
                st.subheader("Aperçu JSON")
                st.json(content)
                mime_type = "application/json"

            elif format_selector == "yaml":
                content = yaml.safe_load(file_content)
                st.subheader("Aperçu YAML")
                st.code(yaml.dump(content, allow_unicode=True), language="yaml")
                mime_type = "application/x-yaml"

            elif format_selector == "md":
                st.subheader("Aperçu Markdown")
                st.markdown(file_content.decode("utf-8"), unsafe_allow_html=True)
                mime_type = "text/markdown"

            elif format_selector == "txt":
                st.subheader("Aperçu Texte")
                st.text_area("Contenu texte", file_content.decode("utf-8"), height=400)
                mime_type = "text/plain"

            # Bouton de téléchargement du fichier converti
            st.download_button(
                label=f"Télécharger le fichier {format_selector.upper()}",
                data=file_content,
                file_name=output_file_key,
                mime=mime_type
            )

        except Exception as e:
            st.error(f"Erreur lors de la récupération du fichier {format_selector.upper()}: {e}")

if __name__ == "__main__":
    main()
