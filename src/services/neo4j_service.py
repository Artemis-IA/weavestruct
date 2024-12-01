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


    def validate_connection(self):
        """
        Validate the connection to the Neo4j database by running a test query.
        """
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                if result.single()[0] == 1:
                    logger.info("Neo4j connection validated successfully.")
        except Exception as e:
            logger.error(f"Failed to validate Neo4j connection: {e}")
            raise


    def create_entity(self, entity: Dict[str, Any]):
        """
        Crée une entité dans Neo4j.
        """
        with self.driver.session() as session:
            return session.write_transaction(self._create_entity, entity)

    @staticmethod
    def _create_entity(tx: Transaction, entity: Dict[str, Any]):
        query = """
        MERGE (e:Entity {id: $id})
        SET e += $properties
        RETURN e
        """
        result = tx.run(query, id=entity["id"], properties=entity.get("properties", {}))
        return result.single()

    def update_entity(self, entity_id: str, properties: Dict[str, Any]):
        """
        Met à jour une entité existante dans Neo4j.
        """
        with self.driver.session() as session:
            return session.write_transaction(self._update_entity, entity_id, properties)

    @staticmethod
    def _update_entity(tx: Transaction, entity_id: str, properties: Dict[str, Any]):
        query = """
        MATCH (e:Entity {id: $id})
        SET e += $properties
        RETURN e
        """
        result = tx.run(query, id=entity_id, properties=properties)
        return result.single()

    def delete_entity(self, entity_id: str):
        """
        Supprime une entité de Neo4j.
        """
        with self.driver.session() as session:
            session.write_transaction(self._delete_entity, entity_id)

    @staticmethod
    def _delete_entity(tx: Transaction, entity_id: str):
        query = """
        MATCH (e:Entity {id: $id})
        DETACH DELETE e
        """
        tx.run(query, id=entity_id)

    def get_entity(self, entity_id: str):
        """
        Récupère une entité spécifique par son ID.
        """
        with self.driver.session() as session:
            return session.read_transaction(self._get_entity, entity_id)

    @staticmethod
    def _get_entity(tx: Transaction, entity_id: str):
        query = """
        MATCH (e:Entity {id: $id})
        RETURN e
        """
        result = tx.run(query, id=entity_id)
        return result.single()

    def get_relationships_by_entity(self, entity_id: str):
        """
        Récupère les relations liées à une entité spécifique.
        """
        with self.driver.session() as session:
            return session.read_transaction(self._get_relationships_by_entity, entity_id)

    @staticmethod
    def _get_relationships_by_entity(tx: Transaction, entity_id: str):
        query = """
        MATCH (e:Entity {id: $id})-[r]->(related)
        RETURN type(r) AS type, properties(r) AS properties, related
        """
        result = tx.run(query, id=entity_id)
        return [{"type": record["type"], "properties": record["properties"], "related": record["related"]} for record in result]