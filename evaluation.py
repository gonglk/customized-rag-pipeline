"""
Implements RAG evaluation utilities using the Ragas library.
"""

import pandas as pd
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    LLMContextRecall,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    Faithfulness,
    FactualCorrectness
)
from ragas.testset.synthesizers.testset_schema import Testset
from langchain.chat_models import ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from tqdm import tqdm
from utils import retry_with_backoff


class RAGEvaluator:
    """
    A helper class to run evaluations on RAG pipelines using the Ragas library.
    """
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initializes a RAG evaluator with the given model name.

        Args:
            model_name: The name of the LLM model to use for metrics that depend on an LLM.
        """
        self.model_name = model_name
        self.llm = LangchainLLMWrapper(ChatOpenAI(model=self.model_name))

    def prepare_evaluation_dataset(self, dataset, qa_chain, rag_type: str) -> EvaluationDataset:
        """
        Prepares the dataset for evaluation by running the QA chain and collecting results.

        Args:
            dataset: Either a ragas.testset.synthesizers.testset_schema.Testset or a pandas DataFrame.
            qa_chain: The pipeline or chain object with an .invoke() method returning answers.
            rag_type: Either 'standard' or 'agent' to specify chain usage.

        Returns:
            An EvaluationDataset object that can be consumed by ragas.evaluate().
        """
        processed_dataset = []

        # Convert input to a DataFrame if needed
        if isinstance(dataset, Testset):
            df = dataset.to_pandas()
        elif isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            raise ValueError("Dataset must be a Testset or a pandas DataFrame")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing evaluation dataset"):
            user_input = row["user_input"]
            reference_text = row["reference"]

            if rag_type == "standard":
                response = qa_chain.invoke({"query": user_input})
                source_docs = response.get("source_documents", [])
                processed_dataset.append({
                    "user_input": user_input,
                    "retrieved_contexts": [doc.page_content for doc in source_docs],
                    "response": response.get("result", ""),
                    "reference": reference_text,
                })
            else:
                # For agentic RAG, we pass a more advanced input schema
                inputs = {
                    "query": user_input,
                    "documents": [],
                    "generation": "",
                    "needs_search": False,
                    "ready_to_answer": False
                }
                response = qa_chain.invoke(inputs)
                docs_list = response.get("documents", [])
                processed_dataset.append({
                    "user_input": user_input,
                    "retrieved_contexts": [doc["doc_content"] for doc in docs_list],
                    "response": response.get("generation", ""),
                    "reference": reference_text,
                })

        return EvaluationDataset.from_list(processed_dataset)

    def run_evaluation(self, dataset, qa_chain, rag_type: str):
        """
        Runs Ragas evaluation on a given dataset and pipeline.

        Args:
            dataset: A test set (DataFrame or Testset).
            qa_chain: The pipeline's chain object implementing .invoke().
            rag_type: 'standard' or 'agent' for specifying how the chain is used.

        Returns:
            A ragas.evaluation.EvaluationResult object containing metric scores.
        """
        evaluation_dataset = self.prepare_evaluation_dataset(dataset, qa_chain, rag_type)
        metrics = [
            LLMContextRecall(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            Faithfulness(),
            FactualCorrectness(),
        ]
        return evaluate(dataset=evaluation_dataset, metrics=metrics, llm=self.llm)

    def generate_testset(self, embeddings: str, chunks, testset_size: int = 20) -> Testset:
        """
        Generates a synthetic test set from provided document chunks using Ragas' TestsetGenerator.

        Args:
            embeddings: The embedding model to use (string name).
            chunks: List of chunked documents to sample from.
            testset_size: Number of QA pairs to generate.

        Returns:
            A ragas.testset.synthesizers.testset_schema.Testset object.
        """
        generator_llm = LangchainLLMWrapper(ChatOpenAI(model=self.model_name))
        generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embeddings))
        generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
        return generator.generate_with_langchain_docs(chunks, testset_size=testset_size)
