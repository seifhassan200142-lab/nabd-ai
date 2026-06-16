import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from .VectorDBInterface import VectorDBInterface

class FaissDBProvider(VectorDBInterface):
    def __init__(self, config, project_id: str = "patient_001"):
        # 1. Inject configuration
        self.config = config
        self.db = None

        # 2. Store the patient/project identifier (no longer hardcoded)
        self.project_id = project_id

        # 3. Initialize the Embedding Model (The engine that converts text to numbers)
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.EMBEDDING_MODEL)

        # 4. Define the storage path dynamically based on the given project_id
        self.store_path = os.path.join(self.config.FILES_DIR, self.project_id, "vector_store")

    def connect(self):
        # FAISS is a local vector store, so connecting means loading the file from disk
        if self.is_collection_existed("default"):
            self.db = FAISS.load_local(
                folder_path=self.store_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.db = None

    def disconnect(self):
        # FAISS does not maintain active network connections, so we just pass
        pass

    def is_collection_existed(self, collection_name: str) -> bool:
        # Check if the FAISS index file actually exists in the target directory
        index_file = os.path.join(self.store_path, "index.faiss")
        return os.path.exists(index_file)

    def create_collection(self, collection_name: str, embedding_size: int, do_reset: bool = False):
        # FAISS creates the collection dynamically during the first insertion, so we pass
        pass

    def insert_many(self, collection_name: str, texts: list, vectors: list = None, metadata: list = None, record_ids: list = None, batch_size: int = 50):
        # Convert texts to vectors and insert them into the database
        if self.db is None:
            self.db = FAISS.from_texts(texts=texts, embedding=self.embeddings, metadatas=metadata)
        else:
            self.db.add_texts(texts=texts, metadatas=metadata)

        # Save the updated index locally
        self.db.save_local(self.store_path)

    def search_by_text(self, collection_name: str, query: str, limit: int = 3) -> List:
        # Ensure the database is loaded before searching
        if not self.db:
            self.connect()

        if not self.db:
            return []

        # Perform similarity search and return the top matching chunks
        results = self.db.similarity_search(query, k=limit)
        return results
    
    