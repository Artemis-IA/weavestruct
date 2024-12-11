from neo4j import GraphDatabase, Transaction
from loguru import logger
from typing import List, Dict, Any, Optional
from src.models.document import Document


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


    def index_graph(self, doc: Document):
        """Process a document to extract nodes and relationships, adding them to Neo4j."""
        try:
            split_docs = self.text_splitter.split_documents([doc])
            split_docs = [
                Document(page_content=self.clean_text(chunk.page_content), metadata=chunk.metadata)
                for chunk in split_docs
            ]
            logger.debug(f"Document split into {len(split_docs)} chunks.")

            # Extract graph data and links
            graph_docs = self.graph_transformer.convert_to_graph_documents(split_docs)
            doc_links = [self.gliner_extractor.extract_many(chunk) for chunk in split_docs]
            # Log extracted nodes, edges, and links
            logger.info("Extracted graph data:")
            for graph_doc in graph_docs:
                if hasattr(graph_doc, "nodes") and graph_doc.nodes:
                    for node in graph_doc.nodes:
                        logger.info(f"Node extracted: ID={node.id}, Name={node.properties.get('name', '')}, Type={node.type}")
                if hasattr(graph_doc, "edges") and graph_doc.edges:
                    for edge in graph_doc.edges:
                        logger.info(f"Edge extracted: Start={edge.start}, End={edge.end}, Type={edge.type}")

            logger.info("Extracted links:")
            for links in doc_links:
                for link in links:
                    logger.info(f"Link extracted: Tag={link.tag}, Kind={link.kind}")

            
            driver = self.neo4j_service.driver
            with driver.session() as session:
                with session.begin_transaction() as tx:
                    for graph_doc, links in zip(graph_docs, doc_links):
                        # Add nodes
                        if hasattr(graph_doc, "nodes") and graph_doc.nodes:
                            for node in graph_doc.nodes:
                                tx.run(
                                    """
                                    MERGE (e:Entity {id: $id, name: $name, type: $type})
                                    ON CREATE SET e.created_at = timestamp()
                                    """,
                                    {
                                        "id": node.id,
                                        "name": node.properties.get("name", ""),
                                        "type": node.type,
                                    },
                                )
                                logger.info(f"Indexed Node: {node.id}, Type: {node.type}")

                        # Add relationships
                        # if hasattr(graph_doc, "edges") and graph_doc.edges:
                        #     GLiRELService.add_relationships(tx, graph_doc.edges)

                        # Add links
                        for link in links:
                            if not link.tag or not link.kind:
                                logger.warning(f"Skipping invalid link: {link}")
                                continue
                            logger.info(f"Adding Link: {link}")
                            tx.run(
                                """
                                MERGE (e:Entity {name: $name})
                                ON CREATE SET e.created_at = timestamp()
                                RETURN e
                                """,
                                {"name": link.tag},
                            )
        except Exception as e:
            logger.error(f"Error processing document: {doc.metadata.get('name', 'unknown')} - {e}")



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