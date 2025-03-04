"""
Configuration constants for Pinecone and external APIs, plus default experiment configs.
"""

# Pinecone Configuration
PINECONE_API_KEY: str = "xx"
INDEX_NAME: str = "challenge-search"
PINECONE_CLOUD: str = "aws"
PINECONE_REGION: str = "us-east-1"

# Services API key 
SERPER_API_KEY: str = "xx"
OPENAI_API_KEY: str = "xx"
MISTRAL_API_KEY: str = "xx"
COHERE_API_KEY: str = "xx"

# Paths
DATA_DIR: str = "data"
EVAL_DIR: str = "evaluation_results"

# Example default experiment configurations (used in main.py or external calls)
CONFIGS = [
    {
        "num_retrievals": 5,
        "retrieval_type": "vector",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-large",
        "llm_provider": "openai",
        "llm_model": "gpt-3.5-turbo",
        "rag_type": "standard"
    },
    {
        "num_retrievals": 5,
        "retrieval_type": "vector",
        "embedding_provider": "cohere",
        "embedding_model": "embed-english-light-v3.0",
        "llm_provider": "openai",
        "llm_model": "gpt-3.5-turbo",
        "rag_type": "standard"
    },
    {
        "num_retrievals": 5,
        "retrieval_type": "hybrid",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-large",
        "llm_provider": "mistralai",
        "llm_model": "mistral-large-latest",
        "rag_type": "standard"
    },
    {
        "num_retrievals": 5,
        "retrieval_type": "hybrid",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-large",
        "llm_provider": "openai",
        "llm_model": "gpt-3.5-turbo",
        "rag_type": "standard"
    },
    {
        "num_retrievals": 5,
        "retrieval_type": "hybrid",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-large",
        "llm_provider": "openai",
        "llm_model": "gpt-3.5-turbo",
        "rag_type": "agent"
    },
    {
        "num_retrievals": 5,
        "retrieval_type": "hybrid",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-large",
        "llm_provider": "mistralai",
        "llm_model": "mistral-large-latest",
        "rag_type": "agent"
    },
    {
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
    }
]
