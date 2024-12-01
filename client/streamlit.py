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

API_URL = "http://localhost:8008/"
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
PROMETHEUS_URL = "http://localhost:8002/metrics"
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

# Configuration MinIO
minio_client = Minio(
    MINIO_ENDPOINT,
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

elif page == "Gestion de Fichiers MinIO":
    st.title("Gestion de Fichiers MinIO")
    buckets = list_buckets()
    selected_bucket = st.selectbox("Choisissez un bucket", [bucket.name for bucket in buckets])

    if selected_bucket:
        objects = list_objects(selected_bucket)
        st.write(f"Fichiers dans le bucket `{selected_bucket}` :")
        for obj in objects:
            st.write(obj.object_name)
        
        uploaded_file = st.file_uploader("Télécharger un fichier dans MinIO")
        if uploaded_file:
            upload_to_minio(selected_bucket, uploaded_file)
            st.success(f"Fichier `{uploaded_file.name}` téléchargé avec succès dans `{selected_bucket}`.")
