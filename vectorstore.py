"""
Initialization for Pinecone and logic to ingest documents into the index.
"""

import os
import time
from uuid import uuid4
from typing import Optional
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings
from config import (
    PINECONE_API_KEY,
    INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    DATA_DIR
)

def add_documents_to_index(
    index,
    embeddings,
    pdf_folder_path: str = DATA_DIR,
    chunk_size: int = 1024,
    chunk_overlap: int = 200
) -> None:
    """
    Loads all PDF documents from a folder, splits them into chunks,
    and inserts them into the given Pinecone index.

    Args:
        index: Pinecone Index object.
        embeddings: Embeddings instance used for vectorization.
        pdf_folder_path (str): Path to the folder containing PDF files.
        chunk_size (int): The size of each text chunk.
        chunk_overlap (int): Overlap between consecutive chunks.

    Returns:
        None
    """
    if not os.path.isdir(pdf_folder_path):
        print(f"Folder not found: {pdf_folder_path}")
        return

    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {pdf_folder_path}; skipping document ingestion.")
        return

    docs = []
    for filename in pdf_files:
        full_path = os.path.join(pdf_folder_path, filename)
        loader = PDFPlumberLoader(full_path)
        docs.extend(loader.load())

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)

    # Convert chunks into vectors and add them to the Pinecone index
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids=uuids)

    print(f"Successfully ingested {len(chunks)} text chunks into the index.")


def initialize_pinecone(
    retries: int = 5,
    delay: int = 5,
    name: str = INDEX_NAME,
    dimension: int = 3072,
    metric: str = "dotproduct"
):
    """
    Initializes and returns a Pinecone index. If the index does not exist, it is created,
    then waits 20 seconds to ensure readiness.

    Args:
        retries (int): Number of retry attempts for creating or accessing the index.
        delay (int): Delay in seconds between each retry attempt.
        name (str): Name of the Pinecone index.
        dimension (int): Dimensionality of the vector space.
        metric (str): Similarity metric to use, e.g. 'dotproduct', 'cosine', etc.

    Raises:
        ValueError: If the provided index name is unrecognized.
        PineconeException: If creation or retrieval fails after retries.

    Returns:
        pinecone.index.Index: The initialized or existing Pinecone index.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Adjust dimension & metric based on known indexes
    if name == "challenge-search":
        dimension = 3072
        metric = "dotproduct"
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    elif name == "challenge-cohere-search":
        dimension = 384
        metric = "cosine"
        embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
    else:
        raise ValueError(
            f"Unrecognized index name: {name}. "
            "Expected one of: 'challenge-search', 'challenge-cohere-search'."
        )

    for attempt in range(retries):
        try:
            # If the index does not exist, create it
            if name not in pc.list_indexes().names():
                print(f"Index '{name}' not found, creating a new one...")
                pc.create_index(
                    name=name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
                )

                print(f"Waiting 20 seconds for index '{name}' to be ready...")
                time.sleep(20)

                index = pc.Index(name)
                add_documents_to_index(index=index, embeddings=embeddings)
                return index
            else:
                print(f"Index '{name}' already exists.")
                return pc.Index(name)

        except PineconeException as e:
            print(f"Attempt {attempt+1}/{retries} failed with error: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("All retries exhausted. Could not initialize Pinecone index.")
                raise e
