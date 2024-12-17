import streamlit as st
import requests
import os
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_agraph import agraph, Node, Edge, Config
import plotly.graph_objects as go

# Charger l'URL de l'API depuis une variable d'environnement
API_URL = os.getenv("API_URL")


def create_interactive_graph_visualization(nodes, relationships):
    """
    Créer une visualisation interactive du graphe avec plusieurs options.
    """
    st.subheader("🔍 Visualisation Détaillée du Graphe")
    
    # Options de visualisation
    viz_option = st.radio("Choisissez le type de visualisation:", 
                           ["Graphe Interactif", "Graphe Networkx", "Graphe Plotly"])
    
    if viz_option == "Graphe Interactif":
        # Configuration advanced pour agraph
        config = Config(
            width=900,
            height=700,
            directed=True,
            nodeHighlightBehavior=True,
            highlightColor="#FABC60",
            collapsible=True,
            node_color="#A3CB38",
            edge_color="#FF6B6B"
        )
        
        # Créer des nodes colorés par type
        node_types = set(node['type'] for node in nodes)
        color_palette = sns.color_palette("husl", len(node_types)).as_hex()
        node_type_color_map = dict(zip(node_types, color_palette))
        
        graph_nodes = [
            Node(
                id=node["name"], 
                label=node["name"], 
                title=f"Type: {node['type']}", 
                size=30,
                color=node_type_color_map.get(node['type'], "#A3CB38")
            ) for node in nodes
        ]
        
        graph_edges = [
            Edge(
                source=rel["source"], 
                target=rel["target"], 
                label=rel["type"],
                width=2
            ) for rel in relationships
        ]
        
        st.markdown("**Graphe interactif avec mise en évidence des relations**")
        agraph(nodes=graph_nodes, edges=graph_edges, config=config)
        
        # Légende des types de nœuds
        st.markdown("##### Légende des Types de Nœuds")
        for node_type, color in node_type_color_map.items():
            st.markdown(f"<span style='background-color:{color}; padding:5px; margin:5px; border-radius:5px;'>{node_type}</span>", unsafe_allow_html=True)
    
    elif viz_option == "Graphe Networkx":
        # Création d'un graphe NetworkX
        G = nx.DiGraph()
        
        # Ajouter les nœuds
        for node in nodes:
            G.add_node(node["name"], type=node["type"])
        
        # Ajouter les relations
        for rel in relationships:
            G.add_edge(rel["source"], rel["target"], type=rel["type"])
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.5)  # positions pour le layout
        node_colors = [plt.cm.Set3(list(G.nodes).index(node)) for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
        nx.draw_networkx_labels(G, pos)
        
        plt.title("Graphe Réseau des Relations")
        plt.axis('off')
        st.pyplot(plt)
    
    else:  # Graphe Plotly
        # Créer un graphe Plotly interactif
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Nœuds',
                    xanchor='left',
                    titleside='right'
                )
            )
        )

        # Création du graphe Plotly
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Graphe Dynamique',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[ dict(
                                text="Graphe de relations",
                                showarrow=False,
                                xref="paper", yref="paper") ],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )

        st.plotly_chart(fig, use_container_width=True)

def display_graph_details(nodes, relationships):
    """
    Afficher des détails supplémentaires sur le graphe
    """
    st.subheader("📊 Statistiques du Graphe")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Nombre de Nœuds", len(nodes))
        
        st.markdown("#### Types de Nœuds")
        node_types = {}
        for node in nodes:
            node_types[node['type']] = node_types.get(node['type'], 0) + 1
        
        for node_type, count in node_types.items():
            st.markdown(f"- {node_type}: {count}")
    
    with col2:
        st.metric("Nombre de Relations", len(relationships))
        
        st.markdown("#### Types de Relations")
        rel_types = {}
        for rel in relationships:
            rel_types[rel['type']] = rel_types.get(rel['type'], 0) + 1
        
        for rel_type, count in rel_types.items():
            st.markdown(f"- {rel_type}: {count}")

def post_upload_graph_visualization(token):
    """
    Récupère et visualise le graphe après upload
    """
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        nodes_response = requests.get(f"{API_URL}/graph", headers=headers)
        
        if nodes_response.status_code == 200:
            graph_data = nodes_response.json()
            nodes = graph_data.get("nodes", [])
            relationships = graph_data.get("edges", [])
            
            if not nodes:
                st.warning("Aucun nœud trouvé dans le graphe.")
                return
            
            create_interactive_graph_visualization(nodes, relationships)
            display_graph_details(nodes, relationships)
        else:
            st.error(f"Erreur lors de la récupération du graphe : {nodes_response.status_code}")
    
    except Exception as e:
        st.error(f"Erreur lors de la visualisation du graphe : {e}")


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

    #                 st.error(f"Erreur : {e}")

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
