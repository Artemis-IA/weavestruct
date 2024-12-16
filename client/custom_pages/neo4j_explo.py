import streamlit as st
from py2neo import Graph
from streamlit_agraph import agraph, Node, Edge, Config
import os

# Load environment variables
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Connect to Neo4j
@st.cache_resource
def connect_to_neo4j(uri, user, password):
    try:
        graph = Graph(uri, auth=(user, password))
        return graph
    except Exception as e:
        st.error(f"Erreur de connexion à Neo4j : {e}")
        return None

graph = connect_to_neo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)


# Helper function to run Cypher queries
def run_cypher_query(query):
    try:
        result = graph.run(query)
        return result.data()
    except Exception as e:
        st.error(f"Erreur lors de l'exécution de la requête : {e}")
        return []


# Helper function to visualize results as graph
def visualize_graph(data):
    nodes, edges = set(), []
    for record in data:
        for key, value in record.items():
            if isinstance(value, dict) and 'id' in value:  # Assuming nodes have `id`
                nodes.add(Node(id=value['id'], label=value.get('label', value['id']), size=20))
            elif isinstance(value, list) and len(value) == 2:  # Assuming relationships are [source, target]
                edges.append(Edge(source=value[0], target=value[1], label=key))
    config = Config(width=800, height=600, directed=True, nodeHighlightBehavior=True)
    agraph(list(nodes), edges, config=config)


# Streamlit app
def main():
    st.title("🔍 Exploration Neo4j avec Cypher")
    if not graph:
        st.warning("Impossible de se connecter à Neo4j. Vérifiez les paramètres d'authentification.")
        return

    st.sidebar.title("⚙️ Options de requête")
    mode = st.sidebar.radio("Mode :", ["Requêtes Préconstruites", "Requête Cypher Personnalisée", "Statistiques du Graphe"])

    if mode == "Requêtes Préconstruites":
        st.subheader("🔗 Requêtes Préconstruites")

        prebuilt_queries = {
            "Lister tous les nœuds": "MATCH (n) RETURN n LIMIT 50",
            "Lister toutes les relations": "MATCH ()-[r]->() RETURN r LIMIT 50",
            "Trouver un chemin court entre deux nœuds": "MATCH (a {id: 'A'}), (b {id: 'B'}), p=shortestPath((a)-[*]-(b)) RETURN p",
            "Statistiques des types de relations": "MATCH ()-[r]->() RETURN type(r) AS Type, count(r) AS Count",
            "Top 10 nœuds avec le plus de relations": "MATCH (n)-[r]->() RETURN n.id AS Node, count(r) AS Degree ORDER BY Degree DESC LIMIT 10"
        }

        query_name = st.selectbox("Sélectionnez une requête :", list(prebuilt_queries.keys()))
        if st.button("Exécuter la requête"):
            query = prebuilt_queries[query_name]
            result = run_cypher_query(query)
            if result:
                st.success(f"Résultat pour : {query_name}")
                st.write(result)
                visualize_graph(result)

    elif mode == "Requête Cypher Personnalisée":
        st.subheader("📝 Requête Cypher Personnalisée")
        custom_query = st.text_area("Écrivez votre requête Cypher :", "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25")
        if st.button("Exécuter la requête personnalisée"):
            result = run_cypher_query(custom_query)
            if result:
                st.success("Résultat de votre requête :")
                st.write(result)
                visualize_graph(result)

    elif mode == "Statistiques du Graphe":
        st.subheader("📊 Statistiques du Graphe")
        st.markdown("""
            **Statistiques disponibles :**
            - Nombre total de nœuds
            - Nombre total de relations
            - Distribution des types de nœuds
            - Distribution des types de relations
        """)

        stats_queries = {
            "Nombre total de nœuds": "MATCH (n) RETURN count(n) AS TotalNodes",
            "Nombre total de relations": "MATCH ()-[r]->() RETURN count(r) AS TotalRelationships",
            "Distribution des types de nœuds": "MATCH (n) RETURN labels(n) AS Labels, count(*) AS Count",
            "Distribution des types de relations": "MATCH ()-[r]->() RETURN type(r) AS Type, count(r) AS Count"
        }

        stats_results = {}
        for title, query in stats_queries.items():
            result = run_cypher_query(query)
            if result:
                stats_results[title] = result

        for title, data in stats_results.items():
            st.markdown(f"### {title}")
            st.write(data)
            if "Count" in data[0]:
                st.bar_chart({row["Labels"] if "Labels" in row else row["Type"]: row["Count"] for row in data})

if __name__ == "__main__":
    main()
