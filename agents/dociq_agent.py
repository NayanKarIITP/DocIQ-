"""
agents/dociq_agent.py
──────────────────────
LangGraph-powered autonomous agent for DocIQ.

Tools available:
  1. search_documents  → searches your Qdrant vector store
  2. web_search        → DuckDuckGo search for recent info
  3. calculate         → safe arithmetic evaluation

The agent decides on its own which tools to call and in what order.
This is the ReAct (Reason + Act) pattern — the industry standard for
agentic AI in 2026.

Why this matters for your resume:
  "Familiarity with agentic frameworks like LangGraph" appears in
  61% of GenAI Engineer JDs in 2026.

Usage:
    from agents.dociq_agent import run_agent
    answer = run_agent("What was the revenue, and how does it compare to inflation?")
"""

from __future__ import annotations
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from loguru import logger

from config import settings


# ── Tool 1: Document search ───────────────────────────────────────────────

@tool
def search_documents(query: str) -> str:
    """
    Search the indexed PDF documents for relevant information.
    Use this for questions about uploaded documents, their content,
    specific facts, tables, or figures mentioned in the documents.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)

        from retrieval.vector_store import VectorStore
        from retrieval.embedder import GeminiEmbedder

        embedder = GeminiEmbedder()
        vs = VectorStore()

        query_emb = embedder.embed_query(query)
        results = vs.search(query_embedding=query_emb, top_k=3)

        if not results:
            return "No relevant documents found in the indexed collection."

        formatted = []
        for r in results:
            formatted.append(
                f"[{r.get('filename','?')} p.{r.get('page_number','?')}] "
                f"{r.get('content','')[:400]}"
            )
        return "\n\n".join(formatted)

    except Exception as e:
        return f"Document search failed: {e}"


# ── Tool 2: Web search ────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """
    Search the web for recent or general information not available
    in the uploaded documents. Use this for:
    - Current events or news
    - Information after 2024
    - General knowledge questions
    - Verifying facts from external sources
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        return result if result else "No web results found."
    except ImportError:
        return (
            "Web search unavailable. Install: "
            "pip install langchain-community duckduckgo-search"
        )
    except Exception as e:
        return f"Web search failed: {e}"


# ── Tool 3: Calculator ────────────────────────────────────────────────────

@tool
def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Use this when the question requires arithmetic, percentages,
    ratios, or any numerical computation.

    Examples:
        "15.2 * 1.18"         → 17.936
        "(500 + 300) / 2"     → 400.0
        "100 * (1 - 0.23)"    → 77.0
    """
    try:
        # Only allow safe math characters
        allowed = set("0123456789+-*/.() eE")
        if not all(c in allowed for c in expression):
            return "Error: only arithmetic expressions allowed"
        result = eval(expression, {"__builtins__": {}}, {})  # safe eval
        return str(round(float(result), 6))
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Calculation error: {e}"


# ── Agent state ───────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


# ── Build the LangGraph agent ─────────────────────────────────────────────

def build_agent():
    """
    Build the LangGraph ReAct agent.

    Graph structure:
        [START] → agent → (tool_call?) → tools → agent → ... → END

    The agent node calls the LLM. If the LLM decides to use a tool,
    the tools node executes it and returns the result to the agent.
    This loops until the LLM produces a final answer (no tool call).
    """
    tools = [search_documents, web_search, calculate]

    # LLM with tools bound — enables tool calling
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.1,
    ).bind_tools(tools)

    # Agent node: calls LLM, may produce tool calls
    def call_agent(state: AgentState) -> dict:
        logger.debug(f"Agent turn — {len(state['messages'])} messages in history")
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_agent)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")

    # Conditional edge: if LLM called a tool → go to tools, else END
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    # After tools always go back to agent
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Public interface ──────────────────────────────────────────────────────

# Build agent once at import time
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def run_agent(question: str, max_steps: int = 10) -> str:
    """
    Run a question through the autonomous agent.

    The agent will:
    1. Decide whether to search documents, web, or calculate
    2. Execute tools as needed
    3. Return a final synthesized answer

    Args:
        question  : user's natural language question
        max_steps : safety limit on agent iterations

    Returns:
        Final answer string from the agent
    """
    logger.info(f"Agent query: {question[:80]}...")

    try:
        agent = get_agent()
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": max_steps},
        )
        # Last message is the final agent response
        final = result["messages"][-1]
        answer = final.content if hasattr(final, "content") else str(final)
        logger.success(f"Agent completed in {len(result['messages'])} turns")
        return answer

    except Exception as e:
        logger.error(f"Agent failed: {e}")
        return f"Agent error: {e}"


def run_agent_with_history(
    question: str,
    history: list[dict] | None = None,
    max_steps: int = 10,
) -> str:
    """
    Multi-turn agent that preserves conversation history.

    Args:
        question : current user question
        history  : list of {"role": "user"|"assistant", "content": ...}
        max_steps: safety limit

    Returns:
        Final answer string
    """
    messages = []

    # Convert history to LangChain message format
    if history:
        for turn in history[-6:]:  # last 3 turns to avoid token overflow
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=question))

    try:
        agent = get_agent()
        result = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": max_steps},
        )
        return result["messages"][-1].content

    except Exception as e:
        logger.error(f"Agent (with history) failed: {e}")
        return f"Agent error: {e}"
