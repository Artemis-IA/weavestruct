import streamlit as st
import os
import requests
from py2neo import Graph
from PyPDF2 import PdfReader
import pandas as pd
import plotly.express as px
from streamlit_agraph import agraph, Node, Edge, Config
from minio import Minio
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
PROMETHEUS_URL = "http://localhost:9090/metrics"
FASTAPI_URL = "http://localhost:8000"
MINIO_API_URL = "localhost:9000"
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

# Configuration MinIO
minio_client = Minio(
    MINIO_API_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

# Streamlit Config
st.set_page_config(page_title="Ultra Mega Streamlit Client", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Pages",
    [
        "Télécharger et Annoter un PDF",
        "Extraction d'Entités & Relations",
        "Visualisation de Graphe",
        "Métriques en Temps Réel",
        "Logs de la Base de Données",
        "Gestion de Fichiers MinIO",
        "Gestion des Modèles ML",
    ],
)

# Helper Functions
def upload_pdf(file):
    files = {"files": (file.name, file.getvalue())}
    response = requests.post(f"{API_URL}/documents/upload/", files=files)
    return response

def fetch_neo4j_data(query):
    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return graph.run(query).data()

def fetch_metrics():
    response = requests.get(PROMETHEUS_URL)
    return response.text if response.status_code == 200 else None

def fetch_logs():
    response = requests.get(f"{API_URL}/logs")
    return response.json() if response.status_code == 200 else []

def list_buckets():
    return minio_client.list_buckets()

def list_objects(bucket_name):
    return minio_client.list_objects(bucket_name)

def upload_to_minio(bucket_name, file):
    minio_client.put_object(bucket_name, file.name, file, length=file.size)

# Pages
if page == "Télécharger et Annoter un PDF":
    st.title("Télécharger et Annoter un PDF")
    
    uploaded_pdf = st.file_uploader("Télécharger un PDF", type=["pdf"])
    if uploaded_pdf:
        with st.spinner("Traitement du PDF..."):
            response = upload_pdf(uploaded_pdf)
            if response.status_code == 200:
                st.success("PDF traité avec succès !")
                data = response.json()

                st.subheader("Annotations")
                st.json(data)

                st.subheader("Contenu Original")
                pdf_reader = PdfReader(uploaded_pdf)
                for i, page in enumerate(pdf_reader.pages):
                    st.markdown(f"### Page {i+1}")
                    st.text(page.extract_text())
            else:
                st.error(f"Erreur lors du traitement : {response.status_code} - {response.text}")

elif page == "Extraction d'Entités & Relations":
    st.title("Extraction d'Entités & Relations")
    
    input_text = st.text_area("Entrez un texte pour extraction")
    if st.button("Extraire"):
        with st.spinner("Extraction en cours..."):
            response = requests.post(
                f"{API_URL}/extract-entities-relationships", json={"text": input_text}
            )
            if response.status_code == 200:
                data = response.json()
                st.subheader("Entités")
                st.json(data.get("entities", {}))
                st.subheader("Relations")
                st.json(data.get("relationships", {}))
            else:
                st.error(f"Erreur : {response.status_code} - {response.text}")

elif page == "Visualisation de Graphe":
    st.title("Visualisation de Graphe")
    
    cypher_query = st.text_area("Requête Cypher", "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")
    if st.button("Visualiser"):
        with st.spinner("Récupération des données..."):
            data = fetch_neo4j_data(cypher_query)
            nodes, edges = [], []

            for record in data:
                nodes.append(Node(id=record["n"]["id"], label=record["n"]["label"]))
                edges.append(Edge(source=record["n"]["id"], target=record["m"]["id"], label=record["r"]["type"]))
            
            config = Config(width=800, height=600, directed=True)
            agraph(nodes=nodes, edges=edges, config=config)

elif page == "Métriques en Temps Réel":
    st.title("Métriques en Temps Réel")
    with st.spinner("Récupération des métriques..."):
        metrics = fetch_metrics()
        if metrics:
            lines = metrics.splitlines()
            data = [{"metric": line.split()[0], "value": line.split()[1]} for line in lines if line]
            df = pd.DataFrame(data)
            st.dataframe(df)

            fig = px.line(df, x="metric", y="value", title="Métriques")
            st.plotly_chart(fig)
        else:
            st.warning("Aucune métrique disponible.")

elif page == "Logs de la Base de Données":
    st.title("Logs de la Base de Données")
    logs = fetch_logs()
    if logs:
        df_logs = pd.DataFrame(logs)
        st.dataframe(df_logs)
    else:
        st.warning("Aucun log disponible.")


# Visualisation des Buckets MinIO
elif page == "Visualisation des Buckets MinIO":
    st.header("Visualisation des fichiers dans les buckets MinIO")

    # List available buckets
    buckets = list_buckets()
    BUCKETS = [bucket.name for bucket in buckets]

    selected_bucket = st.selectbox("Choisissez un bucket", BUCKETS)
    if st.button("Lister les fichiers"):
        try:
            # Récupérer les fichiers du bucket sélectionné
            response = requests.get(f"{FASTAPI_URL}/minio/list-bucket/", params={"bucket_name": selected_bucket})
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
                            f"{FASTAPI_URL}/minio/download-url/",
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

elif page == "Gestion des Modèles ML":
    st.title("Gestion des Modèles ML")

    # Authentification
    token = st.text_input("Token d'authentification", type="password")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    # Onglets pour les fonctionnalités
    tabs = st.tabs(["Modèles Disponibles", "Télécharger un Modèle", "Gérer un Modèle"])
    
    # Onglet 1 : Modèles disponibles
    with tabs[0]:
        st.header("Modèles Disponibles")
        if st.button("Charger la liste des modèles"):
            try:
                response = requests.get(f"{API_URL}/loopml/available_models", headers=headers)
                if response.status_code == 200:
                    models = response.json()
                    st.write(models)
                else:
                    st.error(f"Erreur : {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Erreur lors de la récupération des modèles : {str(e)}")

    # Onglet 2 : Télécharger un modèle
    with tabs[1]:
        st.header("Télécharger un Modèle")
        artifact_name = st.text_input("Nom du modèle à télécharger")
        version = st.text_input("Version (optionnel)")
        if st.button("Télécharger le modèle"):
            try:
                params = {"artifact_name": artifact_name}
                if version:
                    params["version"] = version
                response = requests.get(f"{API_URL}/loopml/download_model_artifact", headers=headers, params=params)
                if response.status_code == 200:
                    data = response.json()
                    download_path = data.get("download_path")
                    st.success(f"Modèle téléchargé avec succès : {download_path}")
                else:
                    st.error(f"Erreur : {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Erreur lors du téléchargement : {str(e)}")

    # Onglet 3 : Gestion des modèles
    with tabs[2]:
        st.header("Gestion des Modèles")
        artifact_name = st.text_input("Nom du modèle à supprimer")
        version = st.text_input("Version (optionnel)")
        if st.button("Supprimer le modèle"):
            try:
                data = {"artifact_name": artifact_name, "version": version} if version else {"artifact_name": artifact_name}
                response = requests.delete(f"{API_URL}/loopml/delete_model", headers=headers, json=data)
                if response.status_code == 200:
                    st.success(f"Modèle supprimé avec succès : {artifact_name}")
                else:
                    st.error(f"Erreur : {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Erreur lors de la suppression : {str(e)}")
