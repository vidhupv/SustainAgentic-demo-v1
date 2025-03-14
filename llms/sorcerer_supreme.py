from core.llm.base import BaseLLM
import os
import logging
from core.utils.documents_processor import DocumentProcessor
from core.db.vector_db import ESGVectorDB, KnowhereDB
import ollama

logger = logging.getLogger(__name__)

class SorcererSupremeLLM(BaseLLM):
    """RAG with Multi-DB Routing"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path=config_path)
        self.system_prompt = self._get_system_prompt("config/prompts/sorcerer_supreme.txt")
        self.esg_vector_db = ESGVectorDB(config_path=config_path)
        self.knowhere_db = KnowhereDB(config_path=config_path)

    def index_documents(self, folder: str, use_esg: bool = True):
        """
        Process all text files in a folder and insert their enriched data into
        ESGVectorDB if use_esg is True, or KnowhereDB otherwise.
        """        
        dp = DocumentProcessor(config_path="config/config.yaml")
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) and filename.endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc_type = "esg" if use_esg else "platform_faq"
                enriched_chunks = dp.process_document(content, doc_type=doc_type)
                if use_esg:
                    for chunk in enriched_chunks:
                        self.esg_vector_db.insert_documents(
                            ids=[chunk["metadata"]["chunk_id"]],
                            documents=[chunk["text"]],
                            metadatas=[chunk["metadata"]],
                            embeddings=[chunk["embedding"]]
                        )
                else:
                    for chunk in enriched_chunks:
                        self.knowhere_db.insert_documents(
                            ids=[chunk["metadata"]["chunk_id"]],
                            documents=[chunk["text"]],
                            metadatas=[chunk["metadata"]],
                            embeddings=[chunk["embedding"]]
                        )
                logger.info(f"Indexed file {filename} into {'ESGVectorDB' if use_esg else 'KnowhereDB'}.")

    def answer_query(self, query: str, use_esg: bool = True, k: int = 5) -> dict:
        query_embedding = self.generate_embedding(query, self.embedding_model)
        if use_esg:
            results = self.esg_vector_db.query([query_embedding], n_results=k)
        else:
            results = self.knowhere_db.query([query_embedding], n_results=k)
        context = "\n".join([result["document"] for result in results])
        final_prompt = (
            f"{self.system_prompt}\n\n"
            f"Context:\n{context}\n\n"
            f"User Query: {query}\n\n"
            "Answer the question in detail. If more information is needed, ask follow-up questions."
        )
        answer = self.generate(final_prompt)
        return {"answer": answer, "retrieved": results}
        
    def generate_embedding(self, text: str, model: str) -> list:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]