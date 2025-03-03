"""
Defines a base RAGPipeline class that can be used for both standard (RetrievalQA) and agentic RAG usage.
Initializes an LLM, embeddings, sets up retrieval type, and builds a chain.
"""

import pickle
from typing import Optional
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import ChatMistralAI
from langchain_cohere import CohereEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from pinecone_text.sparse import BM25Encoder
from config import INDEX_NAME
from vectorstore import initialize_pinecone
from template import DEFAULT_TEMPLATE
from utils import preprocess_text


class RAGPipeline:
    """
    Base RAG Pipeline implementation supporting different retrieval and generation strategies.
    """

    def __init__(
        self,
        num_retrievals: int = 5,
        temperature: float = 0.2,
        llm_model: str = "gpt-3.5-turbo",
        llm_provider: str = "openai",
        embedding_model: str = "text-embedding-3-large",
        embedding_provider: str = "openai",
        retrieval_type: str = "hybrid",
        prompt_template: str = DEFAULT_TEMPLATE,
        **kwargs
    ):
        """
        Initialize RAG pipeline with customizable parameters.

        Args:
            chunk_size: Size of each document chunk.
            chunk_overlap: Overlap between consecutive text chunks.
            num_retrievals: Number of documents to retrieve during the query.
            temperature: LLM temperature controlling creativity/randomness.
            llm_model: Model name for the LLM (e.g., 'gpt-3.5-turbo' or 'mistral-large-latest').
            llm_provider: LLM provider name ('openai' or 'mistralai').
            embedding_model: Embedding model name (e.g., 'text-embedding-3-large' or 'embed-english-light-v3.0').
            embedding_provider: Which embedding provider to use ('openai', 'cohere').
            retrieval_type: Type of retrieval ('hybrid' or 'vector').
            prompt_template: The system/human prompt template for final QA.
            **kwargs: Extra arguments that are ignored.
        """
        self.num_retrievals = num_retrievals
        self.temperature = temperature
        self.retrieval_type = retrieval_type
        self.prompt_template = prompt_template

        # Initialize the chosen LLM
        if llm_provider.lower() == "openai":
            self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        elif llm_provider.lower() == "mistralai":
            self.llm = ChatMistralAI(model=llm_model, temperature=temperature)
        else:
            raise ValueError(f"Unsupported llm provider: {llm_provider}")

        # Initialize the chosen embeddings
        if embedding_provider.lower() == "openai":
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
            self.index = initialize_pinecone(name="challenge-search")
        elif embedding_provider.lower() == "cohere":
            self.embeddings = CohereEmbeddings(model=embedding_model)
            self.index = initialize_pinecone(name="challenge-cohere-search")
        else:
            raise ValueError(f"Unsupported embedding provider: {embedding_provider}")

        # Wrap index with a vector store
        self.vector_store = PineconeVectorStore(embedding=self.embeddings, index=self.index)

    def setup_retriever(self):
        """
        Configures the retrieval mechanism based on the specified type.

        Returns:
            A configured retriever object, either PineconeHybridSearchRetriever or a vector-based retriever.
        """
        if self.retrieval_type == "hybrid":
            # Load or train a BM25 sparse encoder
            with open("bm25_encoder.pkl", "rb") as f:
                bm25_encoder = pickle.load(f)

            # Optionally, you could do something like:
            #   corpus = [preprocess_text(doc.page_content) for doc in your_docs]
            #   bm25_encoder.fit(corpus)
            # But we assume 'bm25_encoder.pkl' is already prepared.

            return PineconeHybridSearchRetriever(
                embeddings=self.embeddings,
                sparse_encoder=bm25_encoder,
                index=self.index,
                text_key="text",
                top_k=self.num_retrievals
            )
        else:
            # Vector-based retrieval only
            return self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": self.num_retrievals},
            )

    def setup_rag_chain(self, retriever) -> RetrievalQA:
        """
        Creates a standard retrieval QA chain with the chosen LLM and the prompt template.

        Args:
            retriever: The configured retriever to fetch relevant docs.

        Returns:
            A RetrievalQA chain.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional assistant"),
            ("human", self.prompt_template),
        ])

        return RetrievalQA.from_llm(
            llm=self.llm,
            retriever=retriever,
            prompt=prompt,
            return_source_documents=True
        )

    def build_pipeline(self):
        """
        Orchestrates the creation of a retriever and then a RAG chain.

        Returns:
            A configured QA chain using self.llm and the chosen retrieval method.
        """
        retriever = self.setup_retriever()
        print("retriever set up")
        return self.setup_rag_chain(retriever)

    def run_query(self, query: str, qa_chain) -> dict:
        """
        Executes a query through the RAG pipeline.

        Args:
            query: The user query.
            qa_chain: The chain created by `build_pipeline()`.

        Returns:
            A dictionary with "result" (answer) and "source_documents".
        """
        return qa_chain({"query": query})
