import faiss
import numpy as np
import logging
import yaml

logger = logging.getLogger(__name__)

class FaissConnector:
    """Base class for FAISS vector DB connections."""
    
    def __init__(self, config: dict):
        self.config = config
        self.dimension = config.get("dimension", 768)  # default dimension if not provided
        self.index = self._create_index()
        self.documents = {}  # dictionary to map document IDs to their data
    
    def _create_index(self):
        """Create a FAISS index using IndexFlatL2 (L2 distance)."""
        return faiss.IndexFlatL2(self.dimension)
    
    def insert_documents(self, ids, documents, metadatas, embeddings):
        """Insert documents into the FAISS index and maintain an ID mapping."""
        try:
            embeddings_np = np.array(embeddings).astype("float32")
            self.index.add(embeddings_np)
            for i, doc_id in enumerate(ids):
                self.documents[doc_id] = {
                    "document": documents[i],
                    "metadata": metadatas[i],
                    "embedding": embeddings[i]
                }
            logger.info(f"Inserted {len(ids)} documents into FAISS index.")
        except Exception as e:
            logger.error(f"Insert failed: {str(e)}")
            raise
    
    def query(self, query_embeddings, n_results=5, where=None):
        """Query the FAISS index for nearest neighbors."""
        try:
            query_np = np.array(query_embeddings).astype("float32")
            distances, indices = self.index.search(query_np, n_results)
            results = []
            # Note: This is a simplified mapping. For robust usage, maintain an ordered list of IDs.
            doc_ids = list(self.documents.keys())
            for idx, distance in zip(indices[0], distances[0]):
                # Ensure the index is valid (idx might be -1 if not enough results)
                if idx < 0 or idx >= len(doc_ids):
                    continue
                doc_id = doc_ids[idx]
                results.append({
                    "id": doc_id,
                    "document": self.documents[doc_id]["document"],
                    "metadata": self.documents[doc_id]["metadata"],
                    "distance": float(distance)
                })
            return results
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise
    
    def delete_all_documents(self):
        """Clear the FAISS index and the associated document mapping."""
        try:
            self.index.reset()
            self.documents.clear()
            logger.info("Deleted all documents from FAISS index.")
        except Exception as e:
            logger.error(f"Deletion failed: {str(e)}")
            raise

class KnowhereDB(FaissConnector):
    """Platform-specific knowledge vector store using FAISS."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)['databases']['knowhere_db']
        super().__init__(config)

class ESGVectorDB(FaissConnector):
    """ESG standards vector store using FAISS."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)['databases']['esg_vector_db']
        super().__init__(config)