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
from typing import List
from pathlib import Path

# Charger les variables d'environnement
load_dotenv()

# Configuration des URLs et clés d'API
API_URL = os.getenv("API_URL", "http://localhost:8000")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090/metrics")
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8008")
MINIO_API_URL = os.getenv("MINIO_API_URL", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

# Configuration MinIO
minio_client = Minio(
    MINIO_API_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

# Configuration Streamlit
st.set_page_config(page_title="Integrated Document Processing Hub", layout="wide")

class IntegratedApp:
    def __init__(self):
        self.setup_sidebar()
        self.graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def setup_sidebar(self):
        """Configuration de la sidebar de navigation"""
        st.sidebar.title("📄 Document Hub")
        self.page = st.sidebar.radio(
            "Navigation",
            [
                "📤 Upload & Convert", 
                "📊 Document Analytics", 
                "🔍 Document Search", 
                "⚙️ Export Settings",
                "📂 File Management MinIO",
                "🔧 ML Models Management",
                "🕸️ Graph Visualization",
                "📈 Real-time Metrics",
                "📜 Database Logs",
            ]
        )

    def upload_document(self):
        """Page de téléchargement et conversion de documents"""
        st.header("📤 Document Upload & Conversion")
        
        with st.expander("Supported Formats"):
            st.write("""
            - PDF
            - DOCX
            - PPTX
            - Images
            - HTML
            """)
        
        uploaded_file = st.file_uploader(
            "Téléchargez votre document", 
            type=['pdf', 'docx', 'pptx', 'png', 'jpg', 'jpeg', 'html'],
            help="Formats supportés : PDF, DOCX, PPTX, Images et HTML"
        )
        
        if uploaded_file:
            st.success(f"📂 Document téléchargé : {uploaded_file.name}")
            
            with st.form("export_options"):
                st.subheader("Options d'Export")
                
                export_formats = st.multiselect(
                    "Choisissez les formats d'export",
                    ["JSON", "YAML", "Markdown", "HTML", "CSV"],
                    default=["JSON"]
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    use_ocr = st.checkbox("Activer l'OCR", help="Reconnaissance optique de caractères")
                    export_figures = st.checkbox("Exporter les figures", value=True)
                
                with col2:
                    export_tables = st.checkbox("Exporter les tableaux", value=True)
                    enrich_figures = st.checkbox("Enrichir les figures", help="Ajoute des métadonnées et annotations")
                
                submit_button = st.form_submit_button("🚀 Traiter le document")
            
            if submit_button:
                with st.spinner("Traitement en cours..."):
                    files = {"files": (uploaded_file.name, uploaded_file)}
                    params = {
                        "export_formats": export_formats,
                        "use_ocr": use_ocr,
                        "export_figures": export_figures,
                        "export_tables": export_tables,
                        "enrich_figures": enrich_figures
                    }
                    
                    try:
                        response = requests.post(f"{API_URL}/documents/upload/", files=files, data=params)
                        response.raise_for_status()
                        result = response.json()
                        st.success(f"✅ Document traité : {result}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Erreur de traitement : {e}")

    def document_analytics(self):
        """Page d'analytique des documents"""
        st.header("📊 Document Analytics")
        
        # Simulation de données (à remplacer par une vraie requête API)
        sample_data = {
            "document_type": ["PDF", "DOCX", "PPTX", "Image"],
            "count": [45, 20, 10, 5]
        }
        df = pd.DataFrame(sample_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution des Types de Documents")
            fig1 = px.pie(df, names="document_type", values="count", title="Types de Documents")
            st.plotly_chart(fig1)
        
        with col2:
            st.subheader("Détails")
            st.dataframe(df)

    def document_search(self):
        """Page de recherche de documents"""
        st.header("🔍 Document Search")
        
        search_query = st.text_input("Rechercher un document", placeholder="Mots-clés, titre, etc.")
        search_type = st.selectbox("Type de recherche", ["Texte Intégral", "Métadonnées", "Entités Nommées"])
        
        if st.button("🔎 Rechercher"):
            # Simulation de recherche (remplacer par requête API réelle)
            st.write(f"Recherche de '{search_query}' par {search_type}")

    def export_settings(self):
        """Page de configuration d'export"""
        st.header("⚙️ Export Settings")
        
        storage_options = st.multiselect(
            "Options de stockage",
            ["S3", "MinIO", "Google Cloud Storage", "Local Filesystem"],
            default=["S3"]
        )
        
        export_configs = {
            "JSON": st.checkbox("JSON", value=True),
            "YAML": st.checkbox("YAML"),
            "Markdown": st.checkbox("Markdown"),
            "HTML": st.checkbox("HTML"),
            "CSV": st.checkbox("CSV pour tables")
        }
        
        advanced_options = st.expander("Options Avancées")
        with advanced_options:
            ocr_quality = st.slider("Qualité OCR", 0, 100, 85)
            data_enrichment = st.checkbox("Enrichissement automatique")
        
        if st.button("💾 Sauvegarder Configuration"):
            st.success("Configuration sauvegardée !")

    def file_management_minio(self):
        """Page de gestion des fichiers MinIO"""
        st.header("📂 Gestion des Fichiers MinIO")

        # List available buckets
        try:
            buckets = self.list_buckets()
            BUCKETS = [bucket.name for bucket in buckets]
            if not BUCKETS:
                st.warning("Aucun bucket disponible.")
                return
        except Exception as e:
            st.error(f"Erreur lors de la récupération des buckets : {e}")
            return

        selected_bucket = st.selectbox("Choisissez un bucket", BUCKETS, key="selected_bucket_minio")
        if st.button("Lister les fichiers", key="list_files_minio"):
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
                        selected_file = st.selectbox("Choisissez un fichier à télécharger", [f["Nom du fichier"] for f in file_data], key="selected_file_minio")
                        if st.button("Télécharger", key="download_file_minio"):
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

    def list_buckets(self):
        """Lister les buckets MinIO"""
        return minio_client.list_buckets()

    def list_objects(self, bucket_name):
        """Lister les objets dans un bucket"""
        return minio_client.list_objects(bucket_name)

    def ml_models_management(self):
        """Page de gestion des modèles ML"""
        st.header("🔧 Gestion des Modèles ML")

        # Authentification
        token = st.text_input("Token d'authentification", type="password", key="auth_token_ml")
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        # Onglets pour les fonctionnalités
        tabs = st.tabs(["Modèles Disponibles", "Télécharger un Modèle", "Gérer un Modèle"])
        
        # Onglet 1 : Modèles disponibles
        with tabs[0]:
            st.subheader("Modèles Disponibles")
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
            st.subheader("Télécharger un Modèle")
            artifact_name = st.text_input("Nom du modèle à télécharger", key="download_model_name")
            version = st.text_input("Version (optionnel)", key="download_model_version")
            if st.button("Télécharger le modèle", key="download_model_button"):
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
            st.subheader("Gestion des Modèles")
            artifact_name = st.text_input("Nom du modèle à supprimer", key="delete_model_name")
            version = st.text_input("Version (optionnel)", key="delete_model_version")
            if st.button("Supprimer le modèle", key="delete_model_button"):
                try:
                    data = {"artifact_name": artifact_name, "version": version} if version else {"artifact_name": artifact_name}
                    response = requests.delete(f"{API_URL}/loopml/delete_model", headers=headers, json=data)
                    if response.status_code == 200:
                        st.success(f"Modèle supprimé avec succès : {artifact_name}")
                    else:
                        st.error(f"Erreur : {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Erreur lors de la suppression : {str(e)}")

    def graph_visualization(self):
        """Page de visualisation de graphe avec conversion NL -> Cypher"""
        st.header("🕸️ Visualisation de Graphe")

        cypher_query = st.text_area("Requête Cypher", "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50", key="cypher_query")
        if st.button("Visualiser", key="visualize_cypher"):
            with st.spinner("Récupération des données..."):
                try:
                    data = self.fetch_neo4j_data(cypher_query)
                    nodes, edges = [], []

                    for record in data:
                        nodes.append(Node(id=str(record["n"]["id"]), label=record["n"]["label"]))
                        edges.append(Edge(source=str(record["n"]["id"]), target=str(record["m"]["id"]), label=record["r"]["type"]))
                    
                    config = Config(width=800, height=600, directed=True)
                    agraph(nodes=nodes, edges=edges, config=config)
                except Exception as e:
                    st.error(f"Erreur lors de la visualisation du graphe : {e}")
        
        st.markdown("---")
        st.subheader("🔄 Requête en Langage Naturel")
        natural_query = st.text_input("Entrez votre requête en langage naturel", key="natural_query")
        if st.button("Convertir et Exécuter", key="convert_execute_natural"):
            if natural_query:
                with st.spinner("Conversion de la requête..."):
                    try:
                        cypher_converted = self.convert_natural_to_cypher(natural_query)
                        st.write(f"**Requête Cypher Convertie :** `{cypher_converted}`")
                        # Exécuter la requête convertie
                        data = self.fetch_neo4j_data(cypher_converted)
                        nodes, edges = [], []

                        for record in data:
                            nodes.append(Node(id=str(record["n"]["id"]), label=record["n"]["label"]))
                            edges.append(Edge(source=str(record["n"]["id"]), target=str(record["m"]["id"]), label=record["r"]["type"]))
                        
                        config = Config(width=800, height=600, directed=True)
                        agraph(nodes=nodes, edges=edges, config=config)
                    except Exception as e:
                        st.error(f"Erreur lors de la conversion ou de l'exécution : {e}")
            else:
                st.warning("Veuillez entrer une requête en langage naturel.")

    def convert_natural_to_cypher(self, natural_query: str) -> str:
        """
        Fonction de conversion des requêtes en langage naturel en requêtes Cypher.
        Cette fonction est un exemple simplifié et doit être adaptée selon vos besoins.
        """
        # Exemple simpliste utilisant une correspondance de mots-clés
        # Pour une conversion plus avancée, envisagez d'utiliser un modèle NLP ou une API dédiée.
        natural_query_lower = natural_query.lower()
        if "trouver tous les noeuds" in natural_query_lower:
            return "MATCH (n) RETURN n"
        elif "trouver les relations" in natural_query_lower:
            return "MATCH (n)-[r]->(m) RETURN n, r, m"
        elif "trouver les noeuds de type" in natural_query_lower:
            # Exemple : "trouver les noeuds de type Personne"
            try:
                type_entity = natural_query.split("type")[-1].strip().capitalize()
                return f"MATCH (n:{type_entity}) RETURN n"
            except IndexError:
                raise ValueError("Format de requête non valide pour la conversion.")
        else:
            raise ValueError("La requête en langage naturel n'est pas reconnue ou supportée.")

    def fetch_neo4j_data(self, query: str) -> List[dict]:
        """Exécute une requête Cypher sur Neo4j et retourne les résultats."""
        return self.graph.run(query).data()

    def real_time_metrics(self):
        """Page des métriques en temps réel"""
        st.header("📈 Métriques en Temps Réel")
        with st.spinner("Récupération des métriques..."):
            try:
                metrics = self.fetch_metrics()
                if metrics:
                    lines = metrics.splitlines()
                    data = [{"metric": line.split()[0], "value": line.split()[1]} for line in lines if line]
                    df = pd.DataFrame(data)
                    st.dataframe(df)

                    fig = px.line(df, x="metric", y="value", title="Métriques")
                    st.plotly_chart(fig)
                else:
                    st.warning("Aucune métrique disponible.")
            except Exception as e:
                st.error(f"Erreur lors de la récupération des métriques : {e}")

    def fetch_metrics(self):
        """Récupère les métriques depuis Prometheus"""
        response = requests.get(PROMETHEUS_URL)
        return response.text if response.status_code == 200 else None

    def database_logs(self):
        """Page des logs de la base de données"""
        st.header("📜 Logs de la Base de Données")
        try:
            logs = self.fetch_logs()
            if logs:
                df_logs = pd.DataFrame(logs)
                st.dataframe(df_logs)
            else:
                st.warning("Aucun log disponible.")
        except Exception as e:
            st.error(f"Erreur lors de la récupération des logs : {e}")

    def fetch_logs(self):
        """Récupère les logs depuis l'API"""
        response = requests.get(f"{API_URL}/logs")
        return response.json() if response.status_code == 200 else []

    def run(self):
        """Exécution de l'application"""
        if self.page == "📤 Upload & Convert":
            self.upload_document()
        elif self.page == "📊 Document Analytics":
            self.document_analytics()
        elif self.page == "🔍 Document Search":
            self.document_search()
        elif self.page == "⚙️ Export Settings":
            self.export_settings()
        elif self.page == "📂 File Management MinIO":
            self.file_management_minio()
        elif self.page == "🔧 ML Models Management":
            self.ml_models_management()
        elif self.page == "🕸️ Graph Visualization":
            self.graph_visualization()
        elif self.page == "📈 Real-time Metrics":
            self.real_time_metrics()
        elif self.page == "📜 Database Logs":
            self.database_logs()

def main():
    app = IntegratedApp()
    app.run()

if __name__ == "__main__":
    main()
