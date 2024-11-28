# service/rag_service.py
import os
from typing import Iterable
from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint

class RAGChainService:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = self._initialize_llm()

        # Define the prompt
        self.prompt = PromptTemplate.from_template(
            "Context information is below.\n---------------------\n{context}\n---------------------\n"
            "Given the context information and not prior knowledge, answer the query.\nQuery: {question}\nAnswer:\n"
        )

    def _initialize_llm(self):
        HF_API_KEY = os.environ.get("HF_API_KEY")
        HF_LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        return HuggingFaceEndpoint(
            repo_id=HF_LLM_MODEL_ID,
            huggingfacehub_api_token=HF_API_KEY,
        )

    def format_docs(self, docs: Iterable[LCDocument]):
        """
        Format the documents for RAG.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def build_chain(self):
        """
        Build the RAG chain.
        """
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
