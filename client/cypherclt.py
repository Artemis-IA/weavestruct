import os
import streamlit as st
import networkx as nx
from pyvis.network import Network
from dotenv import load_dotenv
from neo4j import GraphDatabase
from streamlit_agraph import agraph, Node as ANode, Edge as AEdge, Config as AConfig
import pandas as pd

load_dotenv()

st.set_page_config(page_title="Visualisation avancée du Graphe Neo4j", layout="wide")

st.title("🔍 Visualisation avancée du Graphe Neo4j")

# -------------------------------------------------------------------------
# Configuration de la connexion Neo4j via le driver officiel
# -------------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def run_cypher_query(query: str, params: dict = None):
    """Exécute une requête Cypher et retourne la liste des records."""
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query, parameters=params)
        return [r.data() for r in result]

# -------------------------------------------------------------------------
# Sidebar - Paramètres d'affichage et exemples de requêtes
# -------------------------------------------------------------------------
st.sidebar.title("Paramètres d'affichage")

# Sélection du mode de visualisation
viz_mode = st.sidebar.selectbox("Mode de visualisation", ["PyVis", "Streamlit-AGraph"])

# Paramètres communs
physics_enabled = st.sidebar.checkbox("Activer la physique", value=True)
hierarchical_enabled = st.sidebar.checkbox("Vue hiérarchique (AGraph uniquement)", value=False)
directed_edges = st.sidebar.checkbox("Arêtes orientées (AGraph)", value=True)
node_color = st.sidebar.color_picker("Couleur des nœuds", "#00ccff")
node_size = st.sidebar.slider("Taille des nœuds", 5, 50, 20)
edge_color = st.sidebar.color_picker("Couleur des arêtes", "#ffffff")

st.sidebar.markdown("---")

# Exemples de requêtes
st.sidebar.markdown("### Exemples de requêtes")
example_queries = {
    "Top 50 relations": "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50",
    "Tous les nœuds Person": "MATCH (p:Person) RETURN p LIMIT 25",
    "Personnes et leurs relations KNOWS": "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a, r, b LIMIT 30",
    "Compter les types de relations": "MATCH ()-[r]->() RETURN type(r), count(*) as count ORDER BY count DESC",
    "Nœuds sans relations": "MATCH (n) WHERE size((n)--())=0 RETURN n"
}

selected_example = st.sidebar.selectbox("Choisissez une requête exemple", list(example_queries.keys()))
st.sidebar.markdown("Vous pouvez exécuter la requête sélectionnée ou taper la vôtre dans le champ principal.")

# -------------------------------------------------------------------------
# Zone principale : Cheatsheet, saisie et exécution de la requête
# -------------------------------------------------------------------------
with st.expander("📜 Neo4j Cheatsheet"):
    st.markdown("""
    **Basic Setup**
    - `:use <database_name>` : Changer de base
    - `SHOW DATABASES;` : Lister les bases
    - `CALL dbms.components();` : Infos sur l'instance

    **CRUD Nœuds**
    - `CREATE (n:Person {name:"Alice"});`
    - `MATCH (n:Person) RETURN n;`
    - `MATCH (n:Person {name:"Alice"}) SET n.age=30;`
    - `MATCH (n:Person {name:"Alice"}) DELETE n;`

    **CRUD Relations**
    - `MATCH (a:Person {name:"Alice"}),(b:Person {name:"Bob"}) CREATE (a)-[:KNOWS]->(b);`
    - `MATCH (a)-[r:KNOWS]->(b) RETURN a,b,r;`
    - `MATCH (a)-[r:KNOWS]->(b) SET r.since=2023;`
    - `MATCH (a)-[r:KNOWS]->(b) DELETE r;`

    **Filtres & Agrégations**
    - `MATCH (n:Person) WHERE n.age>20 RETURN n;`
    - `MATCH (n:Person) RETURN count(n);`
    """)

st.subheader("Exécuter une requête Cypher")
default_query = example_queries[selected_example]
user_query = st.text_area("Entrez votre requête Cypher :", default_query)

if st.button("Exécuter la requête"):
    try:
        results = run_cypher_query(user_query)
        st.markdown("### Résultats de la requête")
        st.write(f"Nombre de lignes retournées : {len(results)}")

        if len(results) > 0:
            # Affichage tabulaire des résultats
            df = pd.DataFrame(results)
            st.dataframe(df)

            cleaned_results = []
            from neo4j.graph import Node as Neo4jNode, Relationship as Neo4jRel

            G = nx.MultiDiGraph()
            seen_nodes = set()
            seen_rels = set()

            for record in results:
                new_record = {}
                for k, v in record.items():
                    if isinstance(v, Neo4jNode):
                        # On récupère les propriétés du noeud + ses labels
                        # ou simplement une chaîne descriptive
                        labels = list(v.labels)
                        props_str = ", ".join([f"{prop}={val}" for prop, val in dict(v).items()])
                        node_str = f"Node({labels})[{props_str}]"
                        new_record[k] = node_str
                    elif isinstance(v, Neo4jRel):
                        # On récupère le type de la relation et ses propriétés
                        props_str = ", ".join([f"{prop}={val}" for prop, val in dict(v).items()])
                        rel_str = f"Rel({v.type})[{props_str}]"
                        new_record[k] = rel_str
                    else:
                        # Valeur simple
                        new_record[k] = v
                cleaned_results.append(new_record)

            if G.number_of_nodes() > 0:
                st.markdown("### Visualisation du Graphe")

                if viz_mode == "PyVis":
                    # Visualisation PyVis
                    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
                    net.toggle_physics(physics_enabled)

                    for node_id, node_data in G.nodes(data=True):
                        title = "<br>".join([f"{k}: {v}" for k, v in node_data.items() if k != "label"])
                        net.add_node(node_id,
                                     label=node_data.get("label", str(node_id)),
                                     title=title,
                                     shape="dot",
                                     size=node_size,
                                     color=node_color)

                    for source, target, edge_data in G.edges(data=True):
                        rel_label = edge_data.get("label", "")
                        title = "<br>".join([f"{k}: {v}" for k, v in edge_data.items() if k != "label"])
                        net.add_edge(source, target, title=title, label=rel_label, color=edge_color)

                    net.set_options("""
                    var options = {
                      "nodes": {
                        "borderWidth":2,
                        "borderWidthSelected":4,
                        "chosen":true
                      },
                      "edges": {
                        "smooth": {
                          "type": "continuous"
                        },
                        "color": {
                          "inherit": false
                        }
                      },
                      "interaction": {
                        "tooltipDelay": 200,
                        "hideEdgesOnDrag": false
                      }
                    }
                    """)

                    net_html = net.generate_html()
                    st.write("Passez la souris sur un nœud ou une relation pour voir ses propriétés.")
                    st.components.v1.html(net_html, height=800, scrolling=True)

                else:
                    # Visualisation Streamlit-AGraph
                    agraph_nodes = []
                    agraph_edges = []

                    # Convertir le graphe en nœuds et arêtes AGraph
                    for node_id, node_data in G.nodes(data=True):
                        title = "\n".join([f"{k}: {v}" for k, v in node_data.items() if k != "label"])
                        agraph_nodes.append(
                            ANode(
                                id=str(node_id),
                                label=node_data.get("label", str(node_id)),
                                title=title,
                                color=node_color,
                                size=node_size
                            )
                        )

                    for source, target, edge_data in G.edges(data=True):
                        rel_label = edge_data.get("label", "")
                        title = "\n".join([f"{k}: {v}" for k, v in edge_data.items() if k != "label"])
                        agraph_edges.append(
                            AEdge(
                                source=str(source),
                                target=str(target),
                                label=rel_label,
                                title=title,
                                color=edge_color
                            )
                        )

                    config = AConfig(
                        width="100%",
                        height="800px",
                        directed=directed_edges,
                        physics=physics_enabled,
                        hierarchical=hierarchical_enabled,
                    )

                    st.write("Interagissez avec les nœuds et relations. Survolez un élément pour voir ses propriétés.")
                    selected_node = agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)

                    if selected_node:
                        st.markdown("### Détails du nœud sélectionné")
                        # On peut récupérer les propriétés du nœud sélectionné dans G
                        selected_id = int(selected_node) if selected_node.isdigit() else selected_node
                        if selected_id in G.nodes:
                            node_props = G.nodes[selected_id]
                            st.json(node_props)
                        else:
                            st.write("Aucun détail trouvé pour ce nœud.")

        else:
            st.info("Aucun résultat pour cette requête.")

    except Exception as e:
        st.error(f"Erreur lors de l'exécution de la requête : {e}")

# Fermeture du driver
# (streamlit va relancer le script à chaque interaction, donc ce n'est pas strictement nécessaire,
#  mais c'est plus propre)
def close_driver():
    driver.close()


# -------------------------------------------------------------------------
# Informations d'utilisation
# -------------------------------------------------------------------------
st.markdown("""
**Suggestions d'utilisation :**
- Essayez des requêtes plus complexes, explorez vos données.
- Ajustez les couleurs, la taille, et le layout du graphe dans la sidebar.
- Utilisez l'affichage Streamlit-AGraph pour une navigation plus fluide et la sélection de nœuds.
- Consultez le cheatsheet pour vous rappeler de la syntaxe Cypher.
- Combinez filtres et agrégations pour mieux comprendre la structure et le contenu de votre graphe.
""")
