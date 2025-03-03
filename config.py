"""
Configuration constants for Pinecone and external APIs, plus default experiment configs.
"""

# Pinecone Configuration
PINECONE_API_KEY: str = "pcsk_4ZS6GE_N12LyarpEwSHPAGSHYfDCzEJGjW2bNKUuyPzCoZ5hxUf2i7SmPigBdAVzs82hzN"
INDEX_NAME: str = "challenge-search"
PINECONE_CLOUD: str = "aws"
PINECONE_REGION: str = "us-east-1"

# Services API key 
SERPER_API_KEY: str = "c358b815d9f2243ac1b2e542dcdf62e76002bc11"
OPENAI_API_KEY: str = "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"
MISTRAL_API_KEY: str = "sX56mR9pzHO7NYybCssDZ4EJ0MtB9cWo"
COHERE_API_KEY: str = "UQI7sXDk6GHhoxQ2w8JYvrzCGBzhZbsG4hl13Mne"

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
