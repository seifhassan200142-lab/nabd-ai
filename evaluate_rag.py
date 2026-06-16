import sys
import types
import os
import json
import pandas as pd
import warnings

# ==========================================
# 🛠️ Hide annoying deprecation warnings from RAGAS
# ==========================================
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==========================================
# 🛠️ SENIOR HACK: Monkey Patch for RAGAS Bug
# ==========================================
# Temporarily mock the vertexai module to bypass the RAGAS import bug 
# without needing to install unnecessary Google Cloud dependencies.
sys.modules['langchain_community.chat_models.vertexai'] = types.ModuleType('dummy_vertexai')
sys.modules['langchain_community.chat_models.vertexai'].ChatVertexAI = type('ChatVertexAI', (object,), {})
# ==========================================

# Import Langchain integrations
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Import Modern RAGAS (v0.2+) Components and The Big 4 Metrics (excluding AnswerRelevancy for Groq compatibility)
from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics import (
    LLMContextRecall, 
    Faithfulness, 
    ContextPrecision, 
    AnswerCorrectness
)

# Import our custom architecture modules
from config import Config
from src.stores.vectordb.providers.VectorDBProviderFactory import VectorDBProviderFactory
from src.controllers.ChatController import ChatController


def run_evaluation():
    print("[*] Starting the Comprehensive Evaluation Pipeline (4 Metrics)...")

    # Define the path to the evaluation dataset containing ground truth answers
    dataset_path = "eval_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"[-] Error: Evaluation dataset not found at {dataset_path}")
        return

    # Load the evaluation questions and ground truths
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print("[*] Initializing System Components...")
    
    # Initialize the Vector Database using the Factory Pattern
    vector_factory = VectorDBProviderFactory(config=Config)
    vector_db_provider = vector_factory.create(Config.CURRENT_VECTOR_DB)
    
    # Initialize the core Chat Engine
    chat_ctrl = ChatController()

    # Create an empty list to store samples in the modern RAGAS format
    samples = []

    print(f"[*] Processing {len(eval_data)} questions through the RAG system...")

    for item in eval_data:
        question = item["question"]
        ground_truth = item["ground_truth"]

        # 1. Retrieve the relevant medical context from the Vector Database
        retrieved_chunks = vector_db_provider.search_by_text(
            collection_name="default",
            query=question,
            limit=3
        )
        # Extract the pure text content from the Document objects
        contexts = [chunk.page_content for chunk in retrieved_chunks]

        # 2. Generate the answer using our decoupled LLM Engine
        answer = chat_ctrl.generate_answer(query=question, retrieved_chunks=retrieved_chunks)

        # 3. Encapsulate the data into a SingleTurnSample object (Required for RAGAS v0.2+)
        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts,
            response=answer,
            reference=ground_truth
        )
        samples.append(sample)

    print("[+] All questions processed. Configuring RAGAS Evaluator (Judge)...")

    # Convert the collected samples into an EvaluationDataset format for RAGAS
    eval_dataset = EvaluationDataset(samples)

    # Initialize the Groq model to act as the Evaluation Judge
    evaluator_llm = ChatGroq(
        api_key=Config.GROQ_API_KEY, 
        model_name="llama-3.1-8b-instant" 
    )
    
    # Initialize the Embeddings model (Needed specifically for AnswerCorrectness metric)
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2" 
    )

    print("[*] Starting comprehensive RAGAS evaluation using Groq and HuggingFace...")

    # Execute the evaluation pipeline using the modern metric objects
    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            LLMContextRecall(),
            Faithfulness(),
            ContextPrecision(),
            AnswerCorrectness()
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings, # We pass embeddings here because AnswerCorrectness needs them to calculate semantic similarity
    )

    # Display the final results in the terminal
    print("\n" + "="*50)
    print("🏆 Comprehensive Evaluation Results:")
    print("="*50)
    print(result)
    
    # Export the detailed evaluation results to a CSV file for documentation
    df = result.to_pandas()
    df.to_csv("evaluation_results.csv", index=False)
    print("\n[+] Detailed results saved successfully to 'evaluation_results.csv'")

if __name__ == "__main__":
    run_evaluation()

    