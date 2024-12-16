import streamlit as st
import requests
from streamlit_agraph import agraph, Node, Edge, Config
import os

# Charger l'URL de l'API depuis une variable d'environnement
API_URL = os.getenv("API_URL")

def main():
    st.title("🔗 Gestion et Visualisation du Graphe")

    # Vérification du jeton d'authentification
    token = st.session_state.get("token")
    if not token:
        st.warning("Veuillez vous connecter pour accéder à cette page.")
        return

    headers = {"Authorization": f"Bearer {token}"}

    # Onglets
    tabs = st.tabs(["📂 Gestion des Documents", "🔍 Gestion des Entités", "🔗 Gestion des Relations", "🕸️ Visualisation du Graphe"])

    # Onglet 1: Gestion des documents
    with tabs[0]:
        st.header("📂 Gestion des Documents")

        # Choix entre indexation dossier/bucket ou upload PDF
        doc_action = st.radio("Action sur les documents :", ["Indexer depuis dossier/bucket", "Uploader un PDF"])

        if doc_action == "Indexer depuis dossier/bucket":
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

        elif doc_action == "Uploader un PDF":
            uploaded_file = st.file_uploader("Déposez un fichier PDF ici", type=["pdf"])
            if uploaded_file is not None:
                if st.button("Analyser le PDF et construire le graphe"):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                        response = requests.post(f"{API_URL}/graph/upload_pdf", headers=headers, files=files)
                        if response.status_code == 200:
                            st.success("PDF analysé et graphe généré avec succès.")
                        else:
                            st.error(f"Erreur lors de l'analyse du PDF : {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Erreur : {e}")

    # Onglet 2: Gestion des entités (noeuds)
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
                # Envoi en query params car l'API ne spécifie pas de corps JSON
                try:
                    response = requests.post(f"{API_URL}/graph/nodes", headers=headers, params={"name": node_name, "type": node_type})
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

    # Onglet 3: Gestion des relations
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
                # Envoi en query params
                try:
                    response = requests.post(
                        f"{API_URL}/graph/relationships",
                        headers=headers,
                        params={
                            "source_name": source_name,
                            "target_name": target_name,
                            "type": relation_type
                        }
                    )
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
                # Envoi en query params
                try:
                    response = requests.delete(
                        f"{API_URL}/graph/relationships",
                        headers=headers,
                        params={"source_name": source_name, "target_name": target_name}
                    )
                    if response.status_code == 200:
                        st.success("Relation supprimée avec succès.")
                    else:
                        st.error(f"Erreur : {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Erreur : {e}")

    # Onglet 4: Visualisation du graphe
    with tabs[3]:
        st.header("🕸️ Visualisation du Graphe")
        if st.button("Afficher le graphe"):
            try:
                # Récupération des nœuds et relations
                nodes_response = requests.get(f"{API_URL}/graph/nodes", headers=headers)
                relationships_response = requests.get(f"{API_URL}/graph/relationships", headers=headers)

                if nodes_response.status_code == 200 and relationships_response.status_code == 200:
                    nodes = nodes_response.json()
                    relationships = relationships_response.json()

                    # Préparer les données pour la visualisation
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
