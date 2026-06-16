import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ==========================================
# 🛠️ 1. SYSTEM CONFIGURATION & LOGGING
# ==========================================
# Prevent symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Configure the logging system to track server activity and errors
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the decoupled router from our source directory
from src.routes.ChatRoutes import router as ChatRouter

# ==========================================
# 🛠️ 2. FASTAPI APPLICATION INITIALIZATION
# ==========================================
# Initialize the master FastAPI application
app = FastAPI(
    title="Medical-RAG Nabd",
    description="Production-ready Clean Architecture API powered by Mini-RAG methodology.",
    version="1.0.0"
)

# ==========================================
# 🛠️ 3. CORS MIDDLEWARE SETUP
# ==========================================
# This allows external front-end applications (React, Flutter, etc.) 
# to communicate with our API without browser security blocks.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing (Change to specific domains in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Register/Include the Chat Router into our main application
app.include_router(ChatRouter)

@app.get("/")
def system_health():
    """Root endpoint for infrastructure monitoring."""
    logger.info("Health check endpoint accessed.")
    return {"status": "Online", "architecture": "Decoupled Mini-RAG"}

# Uvicorn entrypoint execution
if __name__ == "__main__":
    logger.info("[*] Launching decoupled FastAPI application server...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

