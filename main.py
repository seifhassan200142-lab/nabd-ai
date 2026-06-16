import os
# Suppress native Windows environment symbolic linkage warning traps programmatically
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import torch 
import json
import shutil

from config import Config
from src.controllers.DataController import DataController
from src.controllers.ProcessController import ProcessController
from src.controllers.ChatController import ChatController

# Updated import path matching your IDE's current structure
from src.stores.vectordb.providers.VectorDBProviderFactory import VectorDBProviderFactory
from langchain_core.documents import Document

def test_system():
    print("[*] Welcome to the Medical-RAG Smart Hospital System...")
    
    # ==========================================
    # Initialization & Routing
    # ==========================================
    original_name = "medical_combined.json" 
    source_file_path = os.path.join(Config.DATA_DIR, original_name)
    patient_id = "patient_001" 
    
    if not os.path.exists(source_file_path):
        print(f"[-] Infrastructure Failure: Target document missing at {source_file_path}")
        return

    # ==========================================
    # Phase 1: Data Ingestion 
    # ==========================================
    data_ctrl = DataController()
    print(f"[*] Ingesting file and creating secure archive for {patient_id}...")
    new_safe_path, final_file_name = data_ctrl.generate_unique_filepath(
        orig_file_name=original_name, 
        project_id=patient_id
    )
    shutil.copy(source_file_path, new_safe_path)
    print(f"[+] File successfully archived at: {new_safe_path}")
    
    # ==========================================
    # Phase 2 & 3: Dynamic Parsing & Intelligent Chunking
    # ==========================================
    file_extension = os.path.splitext(original_name)[1].lower()
    chunks = []

    if file_extension == '.json':
        print("\n[*] Initializing JSON parsing for structured data...")
        with open(new_safe_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Core logic to convert JSON objects into Document chunks
        if isinstance(data, list):
            for i, item in enumerate(data):
                content = json.dumps(item, ensure_ascii=False)
                chunks.append(Document(page_content=content, metadata={"source": original_name, "chunk_id": i}))
        elif isinstance(data, dict):
            for k, v in data.items():
                content = f"{k}: {json.dumps(v, ensure_ascii=False)}"
                chunks.append(Document(page_content=content, metadata={"source": original_name}))
        print(f"[+] Successfully loaded and chunked {len(chunks)} elements from JSON.")

    elif file_extension == '.csv':
        print("\n[*] Initializing CSVLoader for structured data parsing...")
        from langchain_community.document_loaders.csv_loader import CSVLoader
        loader = CSVLoader(file_path=new_safe_path, encoding='utf-8')
        chunks = loader.load()
        print(f"[+] Successfully loaded and chunked {len(chunks)} rows from CSV.")

    elif file_extension == '.pdf':
        print("\n[*] Initializing Medical VLM (Docling) for multimodal parsing...")
        process_ctrl = ProcessController(project_id=patient_id)
        medical_report_md = process_ctrl.process_medical_document(file_name=final_file_name)
        
        if medical_report_md:
            print("\n[+] Parsing successful! Proceeding to Chunking...")
            chunks = process_ctrl.chunk_medical_document(markdown_text=medical_report_md)
    else:
        print(f"[-] Unsupported file format: {file_extension}")
        return

    # ==========================================
    # Exporting Chunks & Proceeding to DB
    # ==========================================
    if chunks:
        # 1. Export JSON Logs for Debugging
        project_dir = os.path.dirname(new_safe_path)
        export_path = os.path.join(project_dir, "medical_chunks_export.json")
        
        all_chunks_data = [
            {"chunk_id": getattr(c, 'metadata', {}).get('chunk_id', i), 
             "metadata": c.metadata, 
             "content": c.page_content} 
            for i, c in enumerate(chunks)
        ]
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunks_data, f, indent=4, ensure_ascii=False)
        print(f"[+] {len(chunks)} chunks exported to JSON mapping logs.")
        
        # ==========================================
        # ⚠️ TESTING MODE: Slicing data for fast execution
        # ==========================================
        chunks = chunks[:100]
        print(f"\n[*] Testing Mode Activated: Sliced down to {len(chunks)} chunks for quick processing!")

        # ==========================================
        # Phase 4: Text Embeddings & Vector Storage (Factory)
        # ==========================================
        print("\n" + "=" * 60)
        print("Phase 4: Generating Embeddings & Building Vector DB")
        print("=" * 60)
        
        vector_factory = VectorDBProviderFactory(config=Config)
        vector_db_provider = vector_factory.create(Config.CURRENT_VECTOR_DB)
        
        if vector_db_provider is None:
            raise ValueError(f"Unsupported Vector DB provider: {Config.CURRENT_VECTOR_DB}")

        print(f"[*] Converting {len(chunks)} chunks into Vectors via {Config.CURRENT_VECTOR_DB} Provider...")
        
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        vector_db_provider.insert_many(
            collection_name="default", 
            texts=texts, 
            metadata=metadatas
        )
        print(f"[+] Vector Database successfully saved via Factory.")
        
        # ==========================================
        # Phase 5: Query Execution & Retrieval
        # ==========================================
        print("\n" + "=" * 60)
        print("Phase 5: Testing Vector Memory Index (Retrieval)")
        print("=" * 60)
        
        test_query = "What are the key medical findings or topics discussed in this document?"
        print(f"\n[*] Searching Vector Database for: '{test_query}'...")
        
        retrieved_chunks = vector_db_provider.search_by_text(
            collection_name="default", 
            query=test_query, 
            limit=3
        )
        
        print(f"[+] Found {len(retrieved_chunks)} relevant data chunks.")
        
        # ==========================================
        # Phase 6: Language Generation (LLM Factory)
        # ==========================================
        print("\n" + "=" * 60)
        print("Phase 6: Context-Driven Language Generation (LLM)")
        print("=" * 60)
        
        chat_ctrl = ChatController()
        final_answer = chat_ctrl.generate_answer(query=test_query, retrieved_chunks=retrieved_chunks)
        
        print("\n" + "=" * 60)
        print("🤖 System Context-Aware Response:")
        print("=" * 60)
        print(final_answer)
        print("=" * 60)
            
    else:
        print("[-] Pipeline Error: No chunks were generated from the source file.")

if __name__ == "__main__":
    test_system()

    