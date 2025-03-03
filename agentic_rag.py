"""
Implements an agent-based RAG pipeline using LangGraph.
Agentic flows allow external calls (web search) and document relevance checks.
"""

import os
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.utilities import GoogleSerperAPIWrapper
from config import SERPER_API_KEY
from template import AGENT_RELEVANCE_PROMPT, FINAL_ANSWER_PROMPT, QUERY_REFINER_PROMPT


class AgenticRAGPipeline:
    """
    An agent-based RAG pipeline. Uses a state machine approach to:
    1) Retrieve documents
    2) Verify relevance
    3) Possibly perform web searches if insufficient docs
    4) Generate the final answer
    """
    def __init__(
        self,
        retriever,
        llm_model: str = "gpt-3.5-turbo",
        llm_provider: str = "openai",
        temperature: float = 0.2,
        **kwargs
    ):
        """
        Initialize the Agentic RAG pipeline.

        Args:
            retriever: Document retriever object.
            llm_model: Language model name or instance to use in the agent steps.
            llm_provider: 'openai' or 'mistralai'.
            temperature: LLM temperature for controlling randomness.
            **kwargs: Additional arguments that are ignored.
        """
        self.search_tool = GoogleSerperAPIWrapper()
        self.retriever = retriever

        # Initialize LLM
        if llm_provider.lower() == "openai":
            self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        elif llm_provider.lower() == "mistralai":
            self.llm = ChatMistralAI(model=llm_model, temperature=temperature)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        self._setup_agent_components()
        self._build_graph()

    def _setup_agent_components(self) -> None:
        """Set up various prompt-based chains for verifying relevance, final answer, and query rewriting."""
        # Relevance Checking
        agent_relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", ""),
            ("human", AGENT_RELEVANCE_PROMPT),
        ])
        self.agent_relevance_chain = agent_relevance_prompt | self.llm | JsonOutputParser()

        # Final Answer Generation
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", ""),
            ("human", FINAL_ANSWER_PROMPT),
        ])
        self.final_chain = final_prompt | self.llm | JsonOutputParser()

        # Query Refinement
        query_refiner_prompt = ChatPromptTemplate.from_messages([
            ("system", ""),
            ("human", QUERY_REFINER_PROMPT),
        ])
        self.query_refiner_chain = query_refiner_prompt | self.llm | JsonOutputParser()

    def retrieve_documents(self, state: Dict) -> Dict:
        """Retrieve relevant documents from the vector store."""
        ret_res = self.retriever.invoke(state["query"])
        documents = [{"doc_content": doc.page_content} for doc in ret_res]
        return {"documents": documents}

    def verify_relevance(self, state: Dict) -> Dict:
        """Check if retrieved documents are relevant to the user's query."""
        documents = state["documents"]
        relevant_docs = []

        if not documents:
            # No retrieved documents at all
            return {"needs_search": True}

        for chunk in documents:
            result = self.agent_relevance_chain.invoke({
                "query": state["query"],
                "context": chunk["doc_content"]
            })
            # The chain returns a JSON with "answer" in ["yes","no"]
            if result["answer"].lower() == "yes":
                relevant_docs.append(chunk)

        if not relevant_docs:
            return {"documents": [], "needs_search": True}
        return {"documents": relevant_docs, "ready_to_answer": True}

    def generate_answer(self, state: Dict) -> Dict:
        """
        Generate a final answer using the relevant documents.
        
        Args:
            state: The state dict containing 'documents' and the original 'query'.

        Returns:
            A dict with the 'generation' (final answer) to store in the pipeline state.
        """
        context_list = [doc["doc_content"] for doc in state["documents"]]
        result = self.final_chain.invoke({
            "query": state["query"],
            "context": context_list
        }, config={"max_retries": 3})

        return {"generation": result["answer"]}

    def web_search(self, state: Dict) -> Dict:
        """
        Execute a web search if local retrieved documents are insufficient or irrelevant.
        Merges web search snippets with the original documents for final answer generation.
        """
        documents = state["documents"]
        results = self.search_tool.results(state["query"])
        # Typically results["organic"] is a list of SERP results
        results_list = [{"doc_content": r["snippet"]} for r in results["organic"]]
        documents += results_list
        return {"documents": documents, "ready_to_answer": True}

    def query_rewriter(self, state: Dict) -> Dict:
        """
        Rewrite the user's query for better retrieval, using a specialized prompt chain.
        """
        result = self.query_refiner_chain.invoke({"query": state["query"]}, config={"max_retries": 3})
        return {"query": result["rewrite_query"]}

    def _build_graph(self) -> None:
        """
        Build the LangGraph state machine with the following nodes:
          1) rewrite_query
          2) retrieve
          3) verify
          4) generate
          5) web_search (called if doc retrieval fails)
        """
        class GraphState(TypedDict):
            query: str
            documents: List[Dict]
            generation: str
            needs_search: bool
            ready_to_answer: bool

        workflow = StateGraph(GraphState)

        workflow.add_node("rewrite_query", self.query_rewriter)
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("verify", self.verify_relevance)
        workflow.add_node("generate", self.generate_answer)
        workflow.add_node("web_search", self.web_search)

        # Decide next node based on the 'verify' step's outcome
        def route_decision(state: Dict) -> str:
            if state.get("ready_to_answer", False):
                return "generate"
            if state.get("needs_search", False):
                return "web_search"
            return "generate"

        workflow.add_conditional_edges("verify", route_decision, {
            "web_search": "web_search",
            "generate": "generate"
        })

        # Link the rest of the nodes
        workflow.set_entry_point("rewrite_query")
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("retrieve", "verify")
        workflow.add_edge("web_search", "verify")
        workflow.add_edge("generate", END)

        # Compile the state machine
        self.graph = workflow.compile()

    def run_query(self, query: str) -> Dict:
        """
        Process a query through the agentic RAG pipeline.

        Args:
            query: The user input question.

        Returns:
            A dictionary containing the 'result' (final answer) and 'source_documents'.
        """
        inputs = {
            "query": query,
            "documents": [],
            "generation": "",
            "needs_search": False,
            "ready_to_answer": False
        }
        result = self.graph.invoke(inputs)

        return {
            "result": result["generation"],
            "source_documents": [
                Document(page_content=d["doc_content"]) for d in result["documents"]
            ]
        }
