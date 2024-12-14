import time
import os
import yaml
from fastapi import Form, HTTPException, APIRouter, Depends, Request
from typing import Optional
from neo4j import GraphDatabase
from prometheus_client import Counter, Histogram
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document as LCDocument
from typing import Iterable
from fastapi import APIRouter, HTTPException
from langchain_core.prompts import PromptTemplate
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from loguru import logger
from src.config import settings

router = APIRouter()

REQUEST_LATENCY = Histogram('request_duration_seconds', 'Time spent processing request')
REQUEST_COUNT = Counter('request_total', 'Total number of requests')

ollama_emb = OllamaEmbeddings(model="llama3.2")
llm = ChatOllama(model="llama3.2")

# GLiNER extractor and transformer
with open('conf/gli_config.yml', 'r') as file:
    config = yaml.safe_load(file)
gliner_extractor = GLiNERLinkExtractor(
    labels=config["labels"],
    model="urchade/gliner_multi_pii-v1"
)
# urchade/enrico-two-stage
graph_transformer = GlinerGraphTransformer(
    allowed_nodes=config["allowed_nodes"],
    allowed_relationships=config["allowed_relationships"],
    gliner_model="knowledgator/gliner-multitask-large-v0.5",
    glirel_model="jackboyla/glirel-large-v0",
    entity_confidence_threshold=0.1,
    relationship_confidence_threshold=0.1,
)

class RAGChainService:
    def __init__(self):
        self.driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
        self.gliner_extractor = GLiNERLinkExtractor(
            labels=config["labels"],
            model="E3-JSI/gliner-multi-pii-domains-v1"
        )
        self.graph_transformer = GlinerGraphTransformer(
            allowed_nodes=config["allowed_nodes"],
            allowed_relationships=config["allowed_relationships"],
            gliner_model="knowledgator/gliner-multitask-large-v0.5",
            glirel_model="jackboyla/glirel-large-v0",
            entity_confidence_threshold=0.1,
            relationship_confidence_threshold=0.1,
        )
        self.retriever = Neo4jVector.from_existing_index(
            ollama_emb,
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
            index_name="vector_index",
            keyword_index_name="keyword",
            search_type="hybrid",
        )
        self.llm = self._initialize_llm()

        # Define the prompt
        self.prompt = PromptTemplate.from_template(
            "Context information is below.\n---------------------\n{context}\n---------------------\n"
            "Given the context information and not prior knowledge, answer the query.\nQuery: {question}\nAnswer:\n"
        )
    def format_docs(self, docs: Iterable[LCDocument]):
        return "\n\n".join(doc.page_content for doc in docs)

    def build_chain(self):
        return (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def run_query(self, query: str):
        """
        Run the query through the RAG chain.
        """
        rag_chain = self.build_chain()
        return rag_chain.invoke(query)
    

    def add_graph_to_neo4j(self, graph_docs):
        with self.driver.session() as session:
            for doc in graph_docs:
                for node in doc.nodes:
                    logger.info(f"Adding node: {node.id}, Type: {node.type}")
                    session.run("MERGE (e:Entity {name: $name, type: $type})", {"name": node.id, "type": node.type})

                for edge in doc.relationships:
                    logger.info(f"Adding relationship: {edge.type} between Source: {edge.source.id} and Target: {edge.target.id}")
                    session.run(
                        "MATCH (source:Entity {name: $source}), (target:Entity {name: $target}) "
                        "MERGE (source)-[:RELATED_TO {type: $type}]->(target)",
                        {"source": edge.source.id, "target": edge.target.id, "type": edge.type}
                    )


# Endpoint to index documents
@router.post("/index/")
@REQUEST_LATENCY.time()
def index_pdfs(folder_path: Optional[str] = Form("/home/pi/Documents/IF-SRV/1pdf_subset")):
    REQUEST_COUNT.inc()
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail="Invalid folder path")

    loader = PyPDFDirectoryLoader(folder_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    start_time = time.time()
    for doc in documents:
        if not hasattr(doc, "page_content"):
            continue
        split_docs = text_splitter.split_documents([doc])
        graph_docs = graph_transformer.convert_to_graph_documents(split_docs)
        rag_service = RAGChainService()
        rag_service.add_graph_to_neo4j(graph_docs)
        end_time = time.time()


    return {"message": "Documents indexed successfully"}

# Chat endpoint
@router.post("/chat/")
def chat(query: str):
        store = Neo4jVector.from_existing_index(
            ollama_emb,
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
            index_name="vector",
            keyword_index_name="keyword",
            search_type="hybrid",
        )
        retriever = store.as_retriever()
        service = RAGChainService()
        rag_chain = (
            {"context": retriever.invoke() | service.format_docs, "question": RunnablePassthrough()}
            | PromptTemplate(template="Your question: {question}")
            | llm
            | StrOutputParser()
        )
        response = rag_chain.invoke({"question": query})
        

        return {"query": query, "response": response}
