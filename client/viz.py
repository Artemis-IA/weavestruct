import streamlit as st
import os
import requests
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx
import neo4j
from typing import List, Dict, Any
from pathlib import Path
from langchain_ollama.chat_models import ChatOllama
from typing import List
from pathlib import Path
from minio import Minio
from dotenv import load_dotenv

load_dotenv()

# Configuration
st.set_page_config(page_title="Document Processing Hub", layout="wide")

# Configuration sécurisée avec variables d'environnement
API_URL = os.getenv("API_URL", "http://localhost:8008/")

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
# Styles personnalisés
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
.highlight {
    background-color: #f0f2f6;
    border-radius: 5px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)



class Neo4jIntegration:
    def __init__(self, uri: str, username: str, password: str):
        """
        Initialise la connexion à Neo4j et configure les composants de traduction de requêtes
        """
        try:
            self.driver = neo4j.GraphDatabase.driver(uri, auth=(username, password))
            self.llm = ChatOllama(model="llama3.2")
        except Exception as e:
            st.error(f"Erreur de connexion à Neo4j : {e}")
            self.driver = None

    def convert_nl_to_cypher(self, natural_language_query: str) -> str:
        """
        Convertit une requête en langage naturel en requête Cypher
        """
        prompt = f"""
        Convertis la requête suivante en requête Cypher :
        "{natural_language_query}"

        Règles :
        - Utilise des variables descriptives 
        - Génère une requête Cypher valide
        - Si la requête est ambiguë, fais une supposition raisonnable
        - Utilise MATCH et RETURN
        """
        try:
            cypher_query = self.llm.invoke(prompt).content
            return cypher_query
        except Exception as e:
            st.error(f"Erreur de conversion : {e}")
            return ""

    def execute_cypher_query(self, cypher_query: str) -> List[Dict[str, Any]]:
        """
        Exécute une requête Cypher et retourne les résultats
        """
        if not self.driver:
            st.error("Connexion Neo4j non établie")
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                return [record.data() for record in result]
        except Exception as e:
            st.error(f"Erreur d'exécution Cypher : {e}")
            return []

    def visualize_graph(self, results: List[Dict[str, Any]]):
        """
        Visualise le graphe à partir des résultats de la requête
        """
        G = nx.DiGraph()
        
        for record in results:
            for key, value in record.items():
                G.add_node(key, label=str(value))
                for subkey, subvalue in value.items() if isinstance(value, dict) else {}:
                    G.add_node(subkey, label=str(subvalue))
                    G.add_edge(key, subkey, label=str(subvalue))
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.5)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos, labels={node: data['label'] for node, data in G.nodes(data=True)})
        
        plt.title("Graphe Neo4j")
        plt.axis('off')
        st.pyplot(plt)
        plt.close()

class DocumentVisualizationApp:
    def __init__(self):
        self.setup_sidebar()

    def setup_sidebar(self):
        """Configuration de la sidebar de navigation"""
        st.sidebar.title("📄 Document Hub")
        self.page = st.sidebar.radio(
            "Navigation",
            [
                "📤 Upload & Convert", 
                "📊 Document Analytics", 
                "🔍 Document Search", 
                "🔬 Neo4j Graph", 
                "⚙️ Export Settings"
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

    def neo4j_page(self):
        """Page d'intégration Neo4j"""
        st.header("🔬 Neo4j Knowledge Graph")
        
        # Configuration Neo4j
        col1, col2 = st.columns(2)
        
        with col1:
            uri = st.text_input("URI Neo4j", placeholder="bolt://localhost:7687")
            username = st.text_input("Nom d'utilisateur")
        
        with col2:
            password = st.text_input("Mot de passe", type="password")
            max_records = st.number_input("Nombre max de résultats", min_value=1, max_value=100, value=10)
        
        if st.button("🔒 Connexion"):
            neo4j_integrator = Neo4jIntegration(uri, username, password)
            
            st.subheader("🔍 Requête en Langage Naturel")
            natural_query = st.text_area("Posez votre question en langage naturel")
            
            if st.button("🚀 Traduire & Exécuter"):
                with st.spinner("Conversion de la requête..."):
                    cypher_query = neo4j_integrator.convert_nl_to_cypher(natural_query)
                    st.code(cypher_query, language="cypher")
                
                with st.spinner("Exécution de la requête..."):
                    results = neo4j_integrator.execute_cypher_query(cypher_query)
                    
                    st.subheader("📊 Résultats")
                    st.json(results[:max_records])
                    
                    st.subheader("🌐 Visualisation du Graphe")
                    neo4j_integrator.visualize_graph(results[:max_records])
            
            st.subheader("📝 Exemples de Requêtes")
            example_queries = [
                "Trouve tous les documents liés à un projet spécifique",
                "Liste les documents par type et auteur",
                "Montre les relations entre différents documents"
            ]
            for query in example_queries:
                st.code(query)

    def run(self):
        """Exécution de l'application"""
        if self.page == "📤 Upload & Convert":
            self.upload_document()
        elif self.page == "📊 Document Analytics":
            self.document_analytics()
        elif self.page == "🔍 Document Search":
            self.document_search()
        elif self.page == "🔬 Neo4j Graph":
            self.neo4j_page()
        elif self.page == "⚙️ Export Settings":
            self.export_settings()

def main():
    app = DocumentVisualizationApp()
    app.run()

if __name__ == "__main__":
    main()