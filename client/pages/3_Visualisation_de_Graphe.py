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
