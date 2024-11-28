from neo4j import GraphDatabase, Transaction
from loguru import logger
from typing import List, Dict, Any, Optional


class Neo4jService:
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize the connection to the Neo4j database.
        
        Args:
            uri (str): URI of the Neo4j database.
            user (str): Username for authentication.
            password (str): Password for authentication.
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")

    def close(self):
        """Close the connection to the Neo4j database."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed.")

    def index_graph(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]):
        """
        Index nodes and edges into the Neo4j database.
        """
        with self.driver.session() as session:
            if nodes:
                session.write_transaction(self._index_nodes, nodes)
            if edges:
                session.write_transaction(self._index_edges, edges)

    @staticmethod
    def _index_nodes(tx: Transaction, nodes: List[Dict[str, Any]]):
        """
        Helper function to index nodes into Neo4j.

        Args:
            tx (Transaction): Neo4j transaction object.
            nodes (List[Dict[str, Any]]): List of nodes to index.
        """
        for node in nodes:
            try:
                query = """
                MERGE (n {id: $id})
                ON CREATE SET n += $properties
                """
                tx.run(query, id=node["id"], properties=node.get("properties", {}))
                logger.info(f"Node {node['id']} indexed successfully.")
            except Exception as e:
                logger.error(f"Failed to index node {node['id']}: {e}")

    @staticmethod
    def _index_edges(tx: Transaction, edges: List[Dict[str, Any]]):
        """
        Helper function to index relationships into Neo4j.

        Args:
            tx (Transaction): Neo4j transaction object.
            edges (List[Dict[str, Any]]): List of relationships to index.
        """
        for edge in edges:
            try:
                query = """
                MATCH (a {id: $source}), (b {id: $target})
                MERGE (a)-[r:$type]->(b)
                SET r += $properties
                """
                tx.run(
                    query,
                    source=edge["source"],
                    target=edge["target"],
                    type=edge["type"],
                    properties=edge.get("properties", {}),
                )
                logger.info(f"Edge from {edge['source']} to {edge['target']} indexed successfully.")
            except Exception as e:
                logger.error(f"Failed to index edge from {edge['source']} to {edge['target']}: {e}")

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """
        Retrieve all nodes from the Neo4j database.

        Returns:
            List[Dict[str, Any]]: List of all nodes with their properties.
        """
        with self.driver.session() as session:
            return session.read_transaction(self._get_all_entities_transaction)

    @staticmethod
    def _get_all_entities_transaction(tx: Transaction) -> List[Dict[str, Any]]:
        """
        Helper function to retrieve all nodes.

        Args:
            tx (Transaction): Neo4j transaction object.

        Returns:
            List[Dict[str, Any]]: List of all nodes.
        """
        query = """
        MATCH (e)
        RETURN id(e) AS id, labels(e) AS labels, properties(e) AS properties
        """
        try:
            result = tx.run(query)
            entities = [{"id": record["id"], "labels": record["labels"], "properties": record["properties"]} for record in result]
            return entities
        except Exception as e:
            logger.error(f"Failed to retrieve entities: {e}")
            return []

    def get_all_relationships(self) -> List[Dict[str, Any]]:
        """
        Retrieve all relationships from the Neo4j database.

        Returns:
            List[Dict[str, Any]]: List of all relationships with their properties.
        """
        with self.driver.session() as session:
            return session.read_transaction(self._get_all_relationships_transaction)

    @staticmethod
    def _get_all_relationships_transaction(tx: Transaction) -> List[Dict[str, Any]]:
        """
        Helper function to retrieve all relationships.

        Args:
            tx (Transaction): Neo4j transaction object.

        Returns:
            List[Dict[str, Any]]: List of all relationships.
        """
        query = """
        MATCH ()-[r]->()
        RETURN id(r) AS id, type(r) AS type, startNode(r) AS source, endNode(r) AS target, properties(r) AS properties
        """
        try:
            result = tx.run(query)
            relationships = [
                {
                    "id": record["id"],
                    "type": record["type"],
                    "source": record["source"],
                    "target": record["target"],
                    "properties": record["properties"],
                }
                for record in result
            ]
            return relationships
        except Exception as e:
            logger.error(f"Failed to retrieve relationships: {e}")
            return []

    def generate_graph_visualization(self) -> dict:
        """
        Generate a visualization of the graph by retrieving all nodes and relationships.

        Returns:
            dict: Dictionary containing nodes and relationships.
        """
        with self.driver.session() as session:
            nodes = session.read_transaction(self._get_all_entities_transaction)
            relationships = session.read_transaction(self._get_all_relationships_transaction)
            return {"nodes": nodes, "relationships": relationships}
