from abc import ABC, abstractmethod
from typing import List

class VectorDBInterface(ABC):
    
    # Establish connection to the database
    @abstractmethod
    def connect(self):
        pass

    # Terminate connection safely
    @abstractmethod
    def disconnect(self):
        pass

    # Check if a specific collection/index exists
    @abstractmethod
    def is_collection_existed(self, collection_name: str) -> bool:
        pass

    # Create a new collection or reset an existing one
    @abstractmethod
    def create_collection(self, collection_name: str, embedding_size: int, do_reset: bool = False):
        pass

    # Insert document vectors and their metadata into the database
    @abstractmethod
    def insert_many(self, collection_name: str, texts: list, vectors: list, metadata: list = None, record_ids: list = None, batch_size: int = 50):
        pass
        
    # Search the database using a query text and return the top matching chunks
    @abstractmethod
    def search_by_text(self, collection_name: str, query: str, limit: int = 3) -> List:
        pass

    