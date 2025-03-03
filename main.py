"""
Main entry point for the application. Provides QA pipeline usage and experimentation.
"""

import os
import pandas as pd
import nltk
import time
import json
import warnings


from config import (
    DATA_DIR,
    EVAL_DIR,
    PINECONE_API_KEY,
    SERPER_API_KEY,
    OPENAI_API_KEY,
    MISTRAL_API_KEY,
    COHERE_API_KEY,
    CONFIGS
)
from rag import RAGPipeline
from agentic_rag import AgenticRAGPipeline
from evaluation import RAGEvaluator
from vectorstore import initialize_pinecone
from utils import visualize_evaluation_results

nltk.download("words")
nltk.download("punkt_tab")
os.makedirs(EVAL_DIR, exist_ok=True)

# Set environment variables for LLMs
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["SERPER_API_KEY"] = SERPER_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["COHERE_API_KEY"] = COHERE_API_KEY
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY
warnings.filterwarnings("ignore")

def run_qa_pipeline() -> None:
    """
    Runs an interactive Q&A session based on a custom RAG configuration.
    The user can choose between a 'standard' or 'agentic' pipeline.
    """
    config = build_custom_config()

    if config["rag_type"] == "standard":
        print("Initializing Standard RAG Pipeline...")
        rag_pipeline = RAGPipeline(**config)
        qa_chain = rag_pipeline.build_pipeline()
    else:
        print("Initializing Agentic RAG Pipeline...")
        rag_pipeline = RAGPipeline(**config)
        retriever = rag_pipeline.setup_retriever()
        agentic_rag = AgenticRAGPipeline(
            retriever=retriever,
            llm_provider=config["llm_provider"],
            llm_model=config["llm_model"]
        )

    while True:
        query = input("Enter your question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        if config["rag_type"] == "standard":
            response = qa_chain.invoke({"query": query})
        else:
            response = agentic_rag.run_query(query)

        print("\nResponse:")
        print(response["result"])

def run_experiments() -> None:
    """
    Create multiple RAG pipelines (standard and agentic) with different configurations 
    and compare their performance through an evaluation set.
    """
    print("Select evaluation type:")
    print("1. Running evaluation on default group of RAG pipelines")
    print("2. Running evaluation on your own RAG pipeline")
    eval_type = input("Enter choice (1 or 2): ")

    if eval_type == "1":
        # Default set of configurations
        configs = [{
            "num_retrievals": 5,
            "retrieval_type": "vector",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-large",
            "llm_provider": "openai",
            "llm_model": "gpt-3.5-turbo",
            "rag_type": "agent"
        },
        {
            "num_retrievals": 5,
            "retrieval_type": "vector",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-large",
            "llm_provider": "mistralai",
            "llm_model": "mistral-large-latest",
            "rag_type": "agent"
        }]
        run_config_evaluations(configs)

    elif eval_type == "2":
        # Let user build custom config
        config = build_custom_config()
        run_single_experiment(config)
    else:
        print("Invalid choice. Please try again.")

def build_custom_config() -> dict:
    """
    Interactively construct a config dictionary for RAGPipeline or AgenticRAGPipeline.

    Returns:
        dict: The configuration object with keys:
            - 'retrieval_type': 'hybrid' or 'vector'
            - 'embedding_provider': 'openai' or 'cohere'
            - 'embedding_model': str
            - 'llm_provider': 'openai' or 'mistralai'
            - 'llm_model': str
            - 'rag_type': 'standard' or 'agent'
            - 'num_retrievals': int
    """
    # 1) retrieval type
    print("Select retrieval type:")
    print("1. hybrid")
    print("2. vector")
    r_choice = input("Enter choice (1 or 2): ")
    retrieval_type = "hybrid" if r_choice == "1" else "vector"

    # 2) embedding provider
    if retrieval_type == "hybrid":
        print("Select embedding provider (only openai supported for hybrid in this example):")
        print("1. openai")
        e_choice = input("Enter choice 1: ")
        embedding_provider = "openai"
    else:
        print("Select embedding provider:")
        print("1. openai")
        print("2. cohere")
        e_choice = input("Enter choice (1 or 2): ")
        embedding_provider = "openai" if e_choice == "1" else "cohere"

    # 3) embedding model
    if embedding_provider == "openai":
        print("OpenAI embedding models:\n1. text-embedding-3-large")
        input("Enter choice (default 1): ")  # Only 1 choice here, but user can press Enter
        embedding_model = "text-embedding-3-large"
    else:
        print("Cohere embedding models:\n1. embed-english-light-v3.0")
        input("Enter choice (default 1): ")  # Only 1 choice here, but user can press Enter
        embedding_model = "embed-english-light-v3.0"

    # 4) LLM provider
    print("Select LLM provider:")
    print("1. openai")
    print("2. mistralai")
    l_choice = input("Enter choice (1 or 2): ")
    llm_provider = "openai" if l_choice == "1" else "mistralai"

    # 5) LLM model
    llm_model = "gpt-3.5-turbo"  # default
    if llm_provider == "openai":
        print("OpenAI LLM models:\n1. gpt-3.5-turbo\n2. gpt-4o\n3. gpt-4o-mini")
        model_choice = input("Enter choice (1-3): ")
        if model_choice == "2":
            llm_model = "gpt-4o"
        elif model_choice == "3":
            llm_model = "gpt-4o-mini"
    else:
        print("Mistralai LLM models:\n1. mistral-large-latest")
        model_choice = input("Enter choice (1): ")
        if model_choice == "1":
            llm_model = "mistral-large-latest"
    # 6) temperature
    print("Select temperature:")
    temperature = float(input("Enter from 0 to 1: "))
    
    # 7) num retrievals
    print("Select number of reyrievals:")
    num_retrievals = int(input("Enter from 1 to 10: "))


    # 8) RAG type
    print("Select RAG type:")
    print("1. standard")
    print("2. agent")
    rag_type = "standard" if input("Enter choice (1 or 2): ") == "1" else "agent"

    config = {
        "num_retrievals": num_retrievals,
        "retrieval_type": retrieval_type,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "rag_type": rag_type,
        "temperature": temperature
    }
    print("Final config:", config)
    return config

def run_config_evaluations(configs: list) -> None:
    """
    Run multiple experiments over the provided list of configs 
    and visualize their evaluation results.

    Args:
        configs (list): A list of configuration dictionaries.
    """
    evaluator = RAGEvaluator(model_name="gpt-4o-mini")
    index = initialize_pinecone()  # Might not be used if pipeline calls it
    test_csv = os.path.join(EVAL_DIR, "twenty_testset.csv")

    if not os.path.exists(test_csv):
        print(f"Test set CSV not found at: {test_csv}")
        return

    testset = pd.read_csv(test_csv)

    for config in configs:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        folder_name = f"{config['rag_type']}_rag_eval_{timestamp_str}"
        folder_path = os.path.join(EVAL_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        if config["rag_type"] == "standard":
            print(f"Running standard RAG with config: {config}")
            rag_pipeline = RAGPipeline(**config)
            qa_chain = rag_pipeline.build_pipeline()
            eval_result = evaluator.run_evaluation(testset, qa_chain, "standard")
            csv_file_path = os.path.join(folder_path, "standard_rag_experiment.csv")
        else:
            print(f"Running agentic RAG with config: {config}")
            rag_pipeline = RAGPipeline(**config)
            retriever = rag_pipeline.setup_retriever()
            agentic_rag = AgenticRAGPipeline(retriever=retriever)
            eval_result = evaluator.run_evaluation(testset, agentic_rag.graph, "agent")
            csv_file_path = os.path.join(folder_path, "agentic_rag_experiment.csv")

        # Save config to metadata JSON
        metadata_file = os.path.join(folder_path, "metadata.json")
        with open(metadata_file, "w", encoding="utf-8") as jf:
            json.dump(config, jf, indent=4)

        result_df = eval_result.to_pandas()
        result_df.to_csv(csv_file_path, index=False)
        visualize_evaluation_results(csv_file_path, folder_path)
        print(f"Experiment completed. Results saved to {csv_file_path}\n")

def run_single_experiment(config: dict) -> None:
    """
    Run a single experiment with a user-defined configuration.
    Saves the results and visualizes them.

    Args:
        config (dict): The dictionary describing a custom RAG config.
    """
    evaluator = RAGEvaluator(model_name="gpt-4o-mini")
    index = initialize_pinecone()
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")

    folder_name = f"custom_rag_eval_{config['rag_type']}_{timestamp_str}"
    folder_path = os.path.join(EVAL_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    test_csv = os.path.join(EVAL_DIR, "twenty_testset.csv")
    if not os.path.exists(test_csv):
        print(f"Test set CSV not found at: {test_csv}")
        return

    testset = pd.read_csv(test_csv)
    print(f"\nRunning experiment for custom config: {config}")

    if config["rag_type"] == "standard":
        rag_pipeline = RAGPipeline(**config)
        qa_chain = rag_pipeline.build_pipeline()
        eval_result = evaluator.run_evaluation(testset, qa_chain, "standard")
        csv_file_path = os.path.join(folder_path, "standard_rag_experiment.csv")
    else:
        rag_pipeline = RAGPipeline(**config)
        retriever = rag_pipeline.setup_retriever()
        agentic_rag = AgenticRAGPipeline(retriever=retriever)
        eval_result = evaluator.run_evaluation(testset, agentic_rag.graph, "agent")
        csv_file_path = os.path.join(folder_path, "agentic_rag_experiment.csv")

    # Save config as JSON
    metadata_file = os.path.join(folder_path, "metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as jf:
        json.dump(config, jf, indent=4)

    result_df = eval_result.to_pandas()
    result_df.to_csv(csv_file_path, index=False)
    visualize_evaluation_results(csv_file_path, folder_path)
    print(f"Custom experiment completed. Results saved to {csv_file_path}\n")

if __name__ == "__main__":
    print("Select an option:")
    print("1. Use RAG pipeline for Q&A")
    print("2. Run experiments to compare RAG pipelines")
    choice = input("Enter choice (1 or 2): ")

    if choice == "1":
        run_qa_pipeline()
    elif choice == "2":
        run_experiments()
    else:
        print("Invalid choice. Please restart the script and enter 1 or 2.")
