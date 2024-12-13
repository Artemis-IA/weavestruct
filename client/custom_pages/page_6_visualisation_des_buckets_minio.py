import streamlit as st
import requests
from minio import Minio
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

API_URL = os.getenv("API_URL")
MINIO_API_URL = os.getenv("MINIO_API_URL")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

minio_client = Minio(
    MINIO_API_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

def list_buckets():
    return minio_client.list_buckets()

def main():
    st.title("Visualisation des Buckets MinIO")

    # List available buckets
    buckets = list_buckets()
    BUCKETS = [bucket.name for bucket in buckets]

    selected_bucket = st.selectbox("Choisissez un bucket", BUCKETS)
    if st.button("Lister les fichiers"):
        try:
            response = requests.get(f"{API_URL}/minio/list-bucket/", params={"bucket_name": selected_bucket})
            if response.status_code == 200:
                files = response.json().get("objects", [])
                if files:
                    st.write(f"Fichiers dans le bucket **{selected_bucket}** :")
                    file_data = [
                        {
                            "Nom du fichier": obj["Key"],
                            "Taille (bytes)": obj["Size"],
                            "Dernière modification": obj["LastModified"]
                        }
                        for obj in files
                    ]
                    df_files = pd.DataFrame(file_data)
                    st.dataframe(df_files)

                    # Télécharger un fichier
                    selected_file = st.selectbox("Choisissez un fichier à télécharger", [f["Nom du fichier"] for f in file_data])
                    if st.button("Télécharger"):
                        url_response = requests.get(
                            f"{API_URL}/minio/download-url/",
                            params={"bucket_name": selected_bucket, "object_key": selected_file}
                        )
                        if url_response.status_code == 200:
                            download_url = url_response.json().get("url")
                            st.markdown(f"[Cliquez ici pour télécharger le fichier]({download_url})", unsafe_allow_html=True)
                        else:
                            st.error(f"Erreur lors de la génération de l'URL : {url_response.text}")
                else:
                    st.warning(f"Aucun fichier trouvé dans le bucket **{selected_bucket}**.")
            else:
                st.error(f"Erreur lors de la récupération des fichiers : {response.text}")
        except Exception as e:
            st.error(f"Erreur : {e}")
