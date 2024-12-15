

import streamlit as st
import requests
from streamlit_agraph import agraph, Node, Edge, Config
import os

# Load API URL
API_URL = os.getenv("API_URL")

def main():
    st.title("🔗 Gestion et Visualisation du Graphe")

    # Check user authentication
    token = st.session_state.get("token")
    if not token:
        st.warning("Veuillez vous connecter pour accéder à cette page.")
        return

    headers = {"Authorization": f"Bearer {token}"}

    # Tabs for features
    tabs = st.tabs(["📂 Gestion des Documents", "🔍 Gestion des Entités", "🔗 Gestion des Relations", "🕸️ Visualisation du Graphe"])

    # Tab 1: Manage documents
    with tabs[0]:
        st.header("📂 Gestion des Documents")
        source_type = st.radio("Source des documents :", ["Local", "S3"])
        folder_path, bucket_name, prefix = None, None, None

        if source_type == "Local":
            folder_path = st.text_input("Chemin du dossier local contenant les fichiers PDF")
        else:
            bucket_name = st.text_input("Nom du bucket S3")
            prefix = st.text_input("Préfixe dans le bucket S3 (optionnel)")

        if st.button("Indexer les documents"):
            data = {}
            if source_type == "Local" and folder_path:
                data["folder_path"] = folder_path
            elif source_type == "S3" and bucket_name:
                data["bucket_name"] = bucket_name
                data["prefix"] = prefix

            try:
                response = requests.post(f"{API_URL}/graph/index_nerrel/", headers=headers, data=data)
                if response.status_code == 200:
                    st.success("Documents indexés avec succès.")
                else:
                    st.error(f"Erreur : {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Erreur lors de l'indexation : {e}")

    # Tab 2: Manage entities (nodes)
    with tabs[1]:
        st.header("🔍 Gestion des Entités")
        action = st.radio("Action :", ["Lister tous les nœuds", "Rechercher un nœud", "Créer/Mettre à jour un nœud", "Supprimer un nœud"])

        if action == "Lister tous les nœuds":
            if st.button("Charger les nœuds"):
                try:
                    response = requests.get(f"{API_URL}/graph/nodes", headers=headers)
                    if response.status_code == 200:
                        nodes = response.json()
                        st.write(nodes)
                    else:
                        st.error(f"Erreur : {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Erreur : {e}")

        elif action == "Rechercher un nœud":
            node_name = st.text_input("Nom du nœud à rechercher")
            if st.button("Rechercher"):
                try:
                    response = requests.get(f"{API_URL}/graph/nodes/{node_name}", headers=headers)
                    if response.status_code == 200:
                        node = response.json()
                        st.write(node)
                    else:
                        st.error(f"Erreur : {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Erreur : {e}")

        elif action == "Créer/Mettre à jour un nœud":
            node_name = st.text_input("Nom du nœud")
            node_type = st.text_input("Type du nœud")
            if st.button("Créer/Mettre à jour"):
                try:
                    response = requests.post(f"{API_URL}/graph/nodes", headers=headers, json={"name": node_name, "type": node_type})
                    if response.status_code == 200:
                        st.success("Nœud créé/mis à jour avec succès.")
                    else:
                        st.error(f"Erreur : {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Erreur : {e}")

        elif action == "Supprimer un nœud":
            node_name = st.text_input("Nom du nœud à supprimer")
            if st.button("Supprimer"):
                try:
                    response = requests.delete(f"{API_URL}/graph/nodes/{node_name}", headers=headers)
                    if response.status_code == 200:
                        st.success("Nœud supprimé avec succès.")
                    else:
                        st.error(f"Erreur : {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Erreur : {e}")

    # Tab 3: Manage relationships
    with tabs[2]:
        st.header("🔗 Gestion des Relations")
        action = st.radio("Action :", ["Lister toutes les relations", "Créer une relation", "Supprimer une relation"])

        if action == "Lister toutes les relations":
            if st.button("Charger les relations"):
                try:
                    response = requests.get(f"{API_URL}/graph/relationships", headers=headers)
                    if response.status_code == 200:
                        relationships = response.json()
                        st.write(relationships)
                    else:
                        st.error(f"Erreur : {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Erreur : {e}")

        elif action == "Créer une relation":
            source_name = st.text_input("Nom de la source")
            target_name = st.text_input("Nom de la cible")
            relation_type = st.text_input("Type de relation")
            if st.button("Créer la relation"):
                try:
                    response = requests.post(f"{API_URL}/graph/relationships", headers=headers, json={
                        "source_name": source_name,
                        "target_name": target_name,
                        "type": relation_type
                    })
                    if response.status_code == 200:
                        st.success("Relation créée avec succès.")
                    else:
                        st.error(f"Erreur : {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Erreur : {e}")

        elif action == "Supprimer une relation":
            source_name = st.text_input("Nom de la source")
            target_name = st.text_input("Nom de la cible")
            if st.button("Supprimer la relation"):
                try:
                    response = requests.delete(f"{API_URL}/graph/relationships", headers=headers, json={
                        "source_name": source_name,
                        "target_name": target_name
                    })
                    if response.status_code == 200:
                        st.success("Relation supprimée avec succès.")
                    else:
                        st.error(f"Erreur : {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Erreur : {e}")

    # Tab 4: Graph visualization
    with tabs[3]:
        st.header("🕸️ Visualisation du Graphe")
        if st.button("Afficher le graphe"):
            try:
                # Fetch nodes and relationships
                nodes_response = requests.get(f"{API_URL}/graph/nodes", headers=headers)
                relationships_response = requests.get(f"{API_URL}/graph/relationships", headers=headers)

                if nodes_response.status_code == 200 and relationships_response.status_code == 200:
                    nodes = nodes_response.json()
                    relationships = relationships_response.json()

                    # Prepare nodes and edges for visualization
                    graph_nodes = [Node(id=node["name"], label=node["name"], size=25) for node in nodes]
                    graph_edges = [
                        Edge(source=rel["source"], target=rel["target"], label=rel["type"])
                        for rel in relationships
                    ]

                    config = Config(
                        width=800,
                        height=600,
                        directed=True,
                        nodeHighlightBehavior=True,
                        highlightColor="#F7A7A6",
                        collapsible=True
                    )

                    agraph(nodes=graph_nodes, edges=graph_edges, config=config)
                else:
                    st.error("Erreur lors de la récupération des données du graphe.")
            except Exception as e:
                st.error(f"Erreur : {e}")

if __name__ == "__main__":
    main()


# import streamlit as st
# from py2neo import Graph
# from streamlit_agraph import agraph, Node, Edge, Config
# import os

# NEO4J_URI = os.getenv("NEO4J_URI")
# NEO4J_USER = os.getenv("NEO4J_USER")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# def fetch_neo4j_data(query):
#     graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
#     return graph.run(query).data()

# def main():
#     st.title("Visualisation de Graphe")
#     cypher_query = st.text_area("Requête Cypher", "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")
#     if st.button("Visualiser"):
#         with st.spinner("Récupération des données..."):
#             data = fetch_neo4j_data(cypher_query)
#             nodes, edges = [], []
#             for record in data:
#                 nodes.append(Node(id=record["n"]["id"], label=record["n"]["label"]))
#                 edges.append(Edge(source=record["n"]["id"], target=record["m"]["id"], label=record["r"]["type"]))
#             config = Config(width=800, height=600, directed=True)
#             agraph(nodes=nodes, edges=edges, config=config)
