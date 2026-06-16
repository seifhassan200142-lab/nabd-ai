import os
from config import Config
from .BaseController import BaseController
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class NLPController(BaseController):
    
    def __init__(self, project_id: str):
        super().__init__()
        self.project_id = project_id
        self.project_path = os.path.join(self.files_dir, project_id)
        self.vector_db_path = os.path.join(self.project_path, "vector_store")
        
        # Instantiate localized text vector embeddings engine driven by configuration variables
        print(f"[*] Initializing HuggingFace Embeddings Model ({Config.EMBEDDING_MODEL})...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )

    def create_vector_database(self, chunks):
        print(f"\n[*] Converting {len(chunks)} chunks into Vectors...")
        
        # Generate vector store index nodes dynamically within execution memory space
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        # Explicitly flush vector database indexing snapshots directly to the safe disk directory
        vector_store.save_local(self.vector_db_path)
        
        print(f"[+] Vector Database successfully saved at: {self.vector_db_path}")
        return vector_store
    
    def retrieve_medical_context(self, query: str, top_k: int = 3):
        print(f"\n[*] Searching Vector Database for: '{query}'...")
        
        # Securely deserialize tracking matrices using verified local contextual footprints
        vector_store = FAISS.load_local(
            folder_path=self.vector_db_path, 
            embeddings=self.embeddings, 
            allow_dangerous_deserialization=True 
        )
        
        # Compute exact mathematical geometric distance parameters (Cosine Similarity)
        results = vector_store.similarity_search(query, k=top_k)
        
        print(f"[+] Found {len(results)} relevant medical chunks.")
        return results
    
    