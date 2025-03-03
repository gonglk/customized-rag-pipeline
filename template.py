"""
Prompt templates for standard RAG usage and agentic flows (e.g., verification, final answer, query rewriting).
"""

DEFAULT_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}

Answer:
"""

AGENT_RELEVANCE_PROMPT = """
You are a professional assistant in checking whether chunks are relevant to the Query. 
Decide if the following context is relevant to answer the query.

Query: {query}
Context: {context}

Only answer with "yes" or "no" in the JSON format below:
{{"answer": <"yes" or "no">}}
"""

FINAL_ANSWER_PROMPT = """
You are an assistant helping with questions based on provided documents.
Use the context below to answer the Query. If the context does not contain the answer, say you don't know.

Query: {query}
Context: {context}

Your output must be in json format below.
{{"answer": <"your final answer">}}
"""

QUERY_REFINER_PROMPT = """
Rewrite the user query to be more clear and focused for retrieval.

Query: {query}

Return in the JSON format below:
{{"rewrite_query": <"rewrite query">}}
"""
