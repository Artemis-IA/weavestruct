import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import requests
from py2neo import Graph
import networkx as nx
import matplotlib.pyplot as plt

# Configuration
DATABASE_URL = "postgresql://postgre_user:postgre_password@localhost/postgre_db"
FASTAPI_URL = "http://localhost:8000"  # Adresse de votre serveur FastAPI
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"

# Connexion à PostgreSQL
engine = create_engine(DATABASE_URL)

# Connexion à Neo4j
graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Titre de l'application
st.title("Tableau de bord Document Processing")

# Menu de navigation
menu = st.sidebar.selectbox("Menu", ["Accueil", "Logs PostgreSQL", "Embeddings", "Recherches Similaires", "Neo4j Visualisation", "API FastAPI"])

# Accueil
if menu == "Accueil":
    st.write("Bienvenue sur le tableau de bord Document Processing !")
    st.write("Naviguez via le menu pour explorer différentes fonctionnalités.")

# Logs PostgreSQL
elif menu == "Logs PostgreSQL":
    st.header("Logs des documents")
    query = "SELECT * FROM document_logs"
    try:
        df_logs = pd.read_sql(query, engine)
        st.dataframe(df_logs)
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données : {e}")

# Embeddings
elif menu == "Embeddings":
    st.header("Visualisation des embeddings")
    query = "SELECT * FROM document_embeddings"
    try:
        df_embeddings = pd.read_sql(query, engine)
        st.dataframe(df_embeddings)
    except Exception as e:
        st.error(f"Erreur lors de la récupération des embeddings : {e}")

# Recherches Similaires
elif menu == "Recherches Similaires":
    st.header("Recherche par similarité")
    query = st.text_input("Entrez votre requête")
    top_k = st.number_input("Nombre de résultats", min_value=1, max_value=10, value=5, step=1)
    if st.button("Rechercher"):
        try:
            response = requests.post(f"{FASTAPI_URL}/retrieve_documents/", json={"query": query, "top_k": top_k})
            if response.status_code == 200:
                results = response.json()["results"]
                for res in results:
                    st.write(f"**Contenu :** {res['content']}")
                    st.write(f"**Metadata :** {res['metadata']}")
            else:
                st.error(f"Erreur : {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Erreur lors de la requête : {e}")

# Neo4j Visualisation
elif menu == "Neo4j Visualisation":
    st.header("Visualisation des relations Neo4j")
    try:
        query = "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50"
        data = graph.run(query).data()
        
        # Convertir les données en graphe NetworkX
        G = nx.DiGraph()
        for record in data:
            n = record["n"]
            m = record["m"]
            r = record["r"]
            G.add_edge(n["name"], m["name"], label=r["type"])
        
        # Afficher le graphe
        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "label"))
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Erreur lors de la visualisation : {e}")

# API FastAPI
elif menu == "API FastAPI":
    st.header("Tester les API FastAPI")
    api_choice = st.selectbox("Choisissez une API", ["Logs des documents", "Embeddings", "Recherche"])
    
    if api_choice == "Logs des documents":
        try:
            response = requests.get(f"{FASTAPI_URL}/logs/")
            if response.status_code == 200:
                logs = response.json()
                st.json(logs)
            else:
                st.error(f"Erreur : {response.status_code}")
        except Exception as e:
            st.error(f"Erreur lors de la requête : {e}")
    
    elif api_choice == "Embeddings":
        try:
            response = requests.get(f"{FASTAPI_URL}/embeddings/")
            if response.status_code == 200:
                embeddings = response.json()
                st.json(embeddings)
            else:
                st.error(f"Erreur : {response.status_code}")
        except Exception as e:
            st.error(f"Erreur lors de la requête : {e}")
    
    elif api_choice == "Recherche":
        query = st.text_input("Entrez votre requête")
        top_k = st.number_input("Nombre de résultats", min_value=1, max_value=10, value=5, step=1)
        if st.button("Rechercher"):
            try:
                response = requests.post(f"{FASTAPI_URL}/retrieve_documents/", json={"query": query, "top_k": top_k})
                if response.status_code == 200:
                    results = response.json()["results"]
                    st.json(results)
                else:
                    st.error(f"Erreur : {response.status_code}")
            except Exception as e:
                st.error(f"Erreur lors de la requête : {e}")
