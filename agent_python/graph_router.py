from typing import TypedDict, List, Dict, Any, Callable
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
import logging

logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    input: str
    answer: str
    sources: List[Dict[str, Any]]

def build_graph(
    llm_instance: ChatGoogleGenerativeAI,
    retrieve_top_chunks_fn: Callable[[str, int], List[Dict[str, Any]]],
    web_search_fn: Callable[[str, int], List[Dict[str, Any]]],
    similarity_threshold: float = 0.35,
):
    """Build a simple router graph that chooses between KB RAG and web search.

    Parameters:
        llm_instance: pre-initialized LLM instance with API key
        retrieve_top_chunks_fn: function to retrieve KB chunks with similarity scores
        web_search_fn: function to perform web search
        similarity_threshold: threshold to decide when KB is strong enough
    """
    
    # Configure LLM for concise responses
    llm_instance.temperature = 0.2  # Lower temperature for more focused responses
    
    def decide_tool(state: GraphState) -> str:
        query = state["input"]
        top = retrieve_top_chunks_fn(query, top_k=3)
        best = max((c.get("similarity", 0.0) for c in top), default=0.0)
        state["sources"] = []
        
        # Check if web search is working by doing a test search
        web_test = web_search_fn("test", max_results=1)
        web_available = len(web_test) > 0
        
        # If web search is not available, prefer RAG even with lower similarity
        if not web_available:
            logger.info("Web search unavailable, preferring RAG")
            return "rag"
        
        # Normal decision logic
        return "rag" if best >= similarity_threshold else "tavily"

    def rag_node(state: GraphState) -> GraphState:
        query = state["input"]
        top = retrieve_top_chunks_fn(query, top_k=3)
        ctx = "".join([f"\n[KB {i+1}: {c.get('title','')}]\n{c.get('content','')}\n" for i, c in enumerate(top)])
        prompt = (
            "You are a helpful AI assistant. Use ONLY the knowledge base context provided below to answer the question.\n"
            "IMPORTANT: Be CONCISE and DIRECT. Keep your answer brief and to the point.\n"
            "If the information is not found in the knowledge base, simply say 'Not found in knowledge base.'\n"
            "Do not elaborate unnecessarily or add extra commentary.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {query}\n\n"
            "Answer (be concise):"
        )
        resp = llm_instance.invoke(prompt)
        answer = getattr(resp, "content", None) or str(resp)
        return {"input": query, "answer": answer, "sources": top}

    def tavily_node(state: GraphState) -> GraphState:
        query = state["input"]
        web = web_search_fn(query, max_results=5)  # returns list of dicts
        
        # Check if we have valid web results
        if not web:
            # No web results available, provide a helpful fallback
            answer = "I'm unable to search the web at the moment. Please try again later or ask me about something in my knowledge base."
            return {"input": query, "answer": answer, "sources": []}
        
        web_ctx = ""
        for i, r in enumerate(web):
            web_ctx += f"\n[Web {i+1}: {r.get('title','')}]\n{r.get('content','')}\nURL: {r.get('url','')}\n"
        
        prompt = (
            "You are a helpful AI assistant. Use ONLY the web context provided below to answer the question.\n"
            "IMPORTANT: Be CONCISE and DIRECT. Keep your answer brief and to the point.\n"
            "If the information is not found in the web context, simply say 'Not found on the web.'\n"
            "Do not elaborate unnecessarily or add extra commentary.\n\n"
            f"Web Context:\n{web_ctx}\n\nQuestion: {query}\n\n"
            "Answer (be concise):"
        )
        resp = llm_instance.invoke(prompt)
        answer = getattr(resp, "content", None) or str(resp)
        # Return empty sources for web search to avoid showing Tavily results
        return {"input": query, "answer": answer, "sources": []}

    g = StateGraph(GraphState)
    g.add_node("decide", lambda s: s)
    g.add_node("rag", rag_node)
    g.add_node("tavily", tavily_node)

    g.set_entry_point("decide")
    g.add_conditional_edges(
        "decide",
        decide_tool,
        {"rag": "rag", "tavily": "tavily"},
    )
    g.add_edge("rag", END)
    g.add_edge("tavily", END)
    return g.compile()