
# page_7_gestion_des_modeles_ml.py
#-----
import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL")

def main():
    st.title("Gestion des Modèles ML")
    # Authentification
    with st.sidebar:
        st.subheader("Authentification")
        username = st.text_input("Nom d'utilisateur", key="username")
        password = st.text_input("Mot de passe", type="password", key="password")
        if st.button("Se connecter", key="login_button"):
            try:
                # Requête POST pour s'authentifier
                response = requests.post(
                    f"{API_URL}/auth/token",
                    data={"username": username, "password": password},
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                if response.status_code == 200:
                    token_data = response.json()
                    token = token_data.get("access_token")
                    st.session_state["token"] = token
                    st.session_state["logged_in"] = True
                    st.success("Connexion réussie.")
                else:
                    error_message = response.json().get('detail', 'Erreur d\'authentification.')
                    st.error(f"Erreur : {error_message}")
            except Exception as e:
                st.error(f"Erreur lors de l'authentification : {e}")

    # Vérification du jeton
    token = st.session_state.get("token", None)
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    if not token:
        st.warning("Veuillez vous connecter pour accéder aux fonctionnalités.")
    else:
        # Onglets des fonctionnalités
        tabs = st.tabs(["Liste des Modèles", "Télécharger un Modèle", "Gérer un Modèle"])

        # Onglet 1 : Modèles disponibles
        with tabs[0]:
            st.header("Modèles Disponibles")
            if st.button("Charger la liste des modèles", key="load_models"):
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
            artifact_name = st.text_input("Nom du modèle à télécharger", key="download_name")
            version = st.text_input("Version (optionnel)", key="download_version")
            if st.button("Télécharger le modèle", key="download_model"):
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
            artifact_name = st.text_input("Nom du modèle à supprimer", key="delete_name")
            version = st.text_input("Version (optionnel)", key="delete_version")
            if st.button("Supprimer le modèle", key="delete_model"):
                try:
                    data = {"artifact_name": artifact_name, "version": version} if version else {"artifact_name": artifact_name}
                    response = requests.delete(f"{API_URL}/loopml/delete_model", headers=headers, json=data)
                    if response.status_code == 200:
                        st.success(f"Modèle supprimé avec succès : {artifact_name}")
                    else:
                        st.error(f"Erreur : {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Erreur lors de la suppression : {str(e)}")

#-----

# whole.py
#-----

#-----

# page_3_Visualisation_de_Graphe.py
#-----
import streamlit as st
from py2neo import Graph
from streamlit_agraph import agraph, Node, Edge, Config
import os

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def fetch_neo4j_data(query):
    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return graph.run(query).data()

def main():
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

#-----

# page_2_extraction_d_entites_et_relations.py
#-----
import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL")

def main():
    st.title("Extraction d'Entités & Relation")
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

#-----

# page_1_telecharger_et_annoter_un_pdf.py
#-----
# client/pages/1_Télécharger_et_Annoter_un_PDF.py
import streamlit as st
import requests
from PyPDF2 import PdfReader
import os


def upload_pdf(file):
    files = {"files": (file.name, file.getvalue())}
    API_URL = os.getenv("API_URL")
    response = requests.post(f"{API_URL}/documents/upload/", files=files)
    return response


def main():
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

#-----

# page_4_metriques_en_temps_reel.py
#-----
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL")

def fetch_metrics():
    response = requests.get(PROMETHEUS_URL)
    return response.text if response.status_code == 200 else None

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

#-----

# page_5_logs_de_la_base_de_donnees.py
#-----
import streamlit as st
import requests
import pandas as pd
import os
API_URL = os.getenv("API_URL")

def fetch_logs():
    response = requests.get(f"{API_URL}/logs")
    return response.json() if response.status_code == 200 else []

def main():
    st.title("Logs de la Base de Données")
    logs = fetch_logs()
    if logs:
        df_logs = pd.DataFrame(logs)
        st.dataframe(df_logs)
    else:
        st.warning("Aucun log disponible.")

#-----

# page_6_visualisation_des_buckets_minio.py
#-----
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

#-----
