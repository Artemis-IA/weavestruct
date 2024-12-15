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

# Configuration API et autres
API_URL = os.getenv("API_URL")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL")
MINIO_API_URL = os.getenv("MINIO_API_URL")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

# Configuration MinIO
minio_client = Minio(
    MINIO_API_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "assets", "logo.jpeg")
# Streamlit Config
st.set_page_config(
    page_title="Weavestruct",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://artemis-ia.github.io/weavestruct/',
        'Report a bug': "https://github.com/Artemis-IA/weavestruct/issues",
        'About': "Weavestruct, from docs to intelligence"
    }
)

# Gestion des pages
pages = {
    "📤 Upload & Convert": "custom_pages.page_1_telecharger_et_annoter_un_pdf",
    "Extraction d'Entités & Relations": "custom_pages.page_2_extraction_d_entites_et_relations",
    "🕸️ Visualisation de Graphe": "custom_pages.page_3_visualisation_de_graphe",
    "📈 Métriques en Temps Réel": "custom_pages.page_4_metriques_en_temps_reel",
    "🔍Logs de la Base de Données": "custom_pages.page_5_logs_de_la_base_de_donnees",
    "Gestion de Fichiers MinIO": "custom_pages.page_6_visualisation_des_buckets_minio",
    "📊 Gestion des Modèles ML": "custom_pages.page_7_gestion_des_modeles_ml",
}

# Fonction d'authentification
def authenticate_user(username, password):
    try:
        response = requests.post(
            f"{API_URL}/auth/token",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        if response.status_code == 200:
            return response.json().get("access_token")
        else:
            return None
    except Exception as e:
        st.error(f"Erreur lors de l'authentification : {e}")
        return None

# Authentification obligatoire
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    # Afficher le logo avant connexion
    st.image(logo_path, use_container_width=True)
    st.title("Bienvenue sur Weavestruct")
    st.subheader("Veuillez vous connecter pour continuer.")

    st.sidebar.title("Authentification")
    username = st.sidebar.text_input("Nom d'utilisateur")
    password = st.sidebar.text_input("Mot de passe", type="password")
    if st.sidebar.button("Se connecter"):
        token = authenticate_user(username, password)
        if token:
            st.session_state["authenticated"] = True
            st.session_state["token"] = token
            st.success("Connexion réussie.")
        else:
            st.error("Nom d'utilisateur ou mot de passe incorrect.")

if st.session_state["authenticated"]:
    # Navigation Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Pages", list(pages.keys()))

    # Page Dispatcher
    page_module = __import__(pages[page], fromlist=["main"])
    page_module.main()
else:
    st.warning("Veuillez vous connecter pour accéder à l'application.")

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
