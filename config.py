import os
from dotenv import load_dotenv

# Load sensitive variables from the .env file securely
load_dotenv()

class Config:
    # ==========================================
    # 1. API Keys & Infrastructure Secrets
    # ==========================================
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")

    # ==========================================
    # 2. Directory Architecture
    # ==========================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "Data")
    FILES_DIR = os.path.join(BASE_DIR, "src", "assets")

    # ==========================================
    # 3. Local & Cloud AI Model Specifications
    # ==========================================
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Available LLM engines configuration
    LLM_MODEL_GROQ = "llama-3.1-8b-instant"
    LLM_MODEL_HF = "google/gemma-2-9b-it"

    # Engine Switch: Configured to "GROQ" for ultra-fast and stable execution
    CURRENT_LLM = "GROQ"

    # Controls randomness of the LLM answers. 0.0 = most deterministic/stable
    # (best for a medical assistant). Defined ONCE here only.
    LLM_TEMPERATURE = 0.0

    # Engine Switch for Vector Database
    CURRENT_VECTOR_DB = "FAISS"

    # ==========================================
    # 4. Text Ingestion & Chunking Parameters
    # ==========================================
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    