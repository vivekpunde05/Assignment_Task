"""Phase 2: The Autonomous Content Engine (LangGraph).

Builds a LangGraph state machine that:
  1. Decides a search topic based on a bot persona.
  2. Executes a mock web search.
  3. Drafts a 280-character opinionated post using persona + context.

Output is strictly enforced to JSON: {"bot_id": "...", "topic": "...", "post_content": "..."}
"""
import json
from typing import Annotated, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config import BOT_PERSONAS
from llm_provider import get_llm


# ---------------------------------------------------------------------------
# Mock Tool
# ---------------------------------------------------------------------------
@tool
def mock_searxng_search(query: str) -> str:
    """Simulate a web search returning hardcoded recent headlines based on keywords."""
    q = query.lower()
    if "crypto" in q or "bitcoin" in q or "blockchain" in q:
        return (
            "Recent headlines:\n"
            "- Bitcoin hits new all-time high amid regulatory ETF approvals\n"
            "- Ethereum Layer-2 adoption surges 40% this quarter\n"
            "- SEC approves spot Bitcoin ETFs for major exchanges"
        )
    elif "ai" in q or "openai" in q or "model" in q or "llm" in q:
        return (
            "Recent headlines:\n"
            "- OpenAI releases GPT-4o with multimodal reasoning improvements\n"
            "- EU AI Act enforcement begins next month\n"
            "- Startup claims new inference chip cuts costs by 60%"
        )
    elif "ev" in q or "electric vehicle" in q or "battery" in q:
        return (
            "Recent headlines:\n"
            "- Tesla announces 4680 battery ramp-up in Texas Gigafactory\n"
            "- BYD overtakes Tesla in quarterly global EV sales\n"
            "- Solid-state battery pilot line achieves 1,000 charge cycles"
        )
    elif "market" in q or "stock" in q or "interest rate" in q or "fed" in q:
        return (
            "Recent headlines:\n"
            "- Federal Reserve signals rate cuts in Q3\n"
            "- S&P 500 reaches record high on tech earnings\n"
            "- Bond yields fall as inflation cools to 2.4%"
        )
    elif "space" in q or "mars" in q or "elon" in q or "musk" in q:
        return (
            "Recent headlines:\n"
            "- SpaceX Starship completes successful orbital refueling test\n"
            "- NASA selects lunar rover design for 2026 mission\n"
            "- Private space station module launch planned for 2025"
        )
    elif "privacy" in q or "surveillance" in q or "regulation" in q:
        return (
            "Recent headlines:\n"
            "- New whistleblower reveals social media data-sharing practices\n"
            "- GDPR fines reach €2B total since enforcement began\n"
            "- Congress debates biometric surveillance moratorium"
        )
    else:
        return (
            "Recent headlines:\n"
            "- Global supply chain disruptions ease in Q2\n"
            "- Renewable energy investments surpass fossil fuels for third year\n"
            "- Major tech earnings reports beat analyst expectations"
        )


# ---------------------------------------------------------------------------
# Structured Output Schema
# ---------------------------------------------------------------------------
class BotPostOutput(TypedDict):
    bot_id: str
    topic: str
    post_content: str


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    bot_id: str
    persona: str
    topic: str
    search_query: str
    search_results: str
    post_content: str
    structured_output: dict


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def decide_search(state: GraphState) -> GraphState:
    """Node 1: Given the bot persona, decide what to post about today and build a search query."""
    llm = get_llm(temperature=0.8)
    persona = state["persona"]
    system_prompt = (
        "You are a content strategist for a social media bot. "
        "Given the bot's persona below, decide a single trending topic the bot wants to post about today. "
        "Then output a concise web search query (max 10 words) to gather recent news on that topic.\n\n"
        "Bot Persona:\n"
        f"{persona}\n\n"
        "Respond ONLY with valid JSON in this exact format:\n"
        '{"topic": "the chosen topic", "search_query": "the query string"}'
    )
    response = llm.invoke([SystemMessage(content=system_prompt)])
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback parsing: extract JSON substring
        content = response.content
        start = content.find("{")
        end = content.rfind("}") + 1
        parsed = json.loads(content[start:end])

    state["topic"] = parsed.get("topic", "technology")
    state["search_query"] = parsed.get("search_query", "latest tech news")
    print(f"[Phase 2] Decided topic: '{state['topic']}' | Query: '{state['search_query']}'")
    return state


def web_search(state: GraphState) -> GraphState:
    """Node 2: Execute the mock web search tool."""
    query = state["search_query"]
    result = mock_searxng_search.invoke(query)
    state["search_results"] = result
    print(f"[Phase 2] Search results for '{query}':\n{result}\n")
    return state


def draft_post(state: GraphState) -> GraphState:
    """Node 3: Use persona + search context to draft a highly opinionated, 280-character post."""
    llm = get_llm(temperature=0.9)
    persona = state["persona"]
    topic = state["topic"]
    search_results = state["search_results"]

    system_prompt = (
        "You are a social media bot with a strong, fixed persona. "
        "You MUST stay in character at all times. "
        "Write a highly opinionated, punchy post (max 280 characters) about the topic.\n\n"
        "Bot Persona:\n"
        f"{persona}\n\n"
        "Recent News Context:\n"
        f"{search_results}\n\n"
        "Respond ONLY with valid JSON in this exact format:\n"
        '{"bot_id": "BOT_ID_HERE", "topic": "TOPIC_HERE", "post_content": "POST_TEXT_HERE"}\n\n'
        "Rules:\n"
        "- post_content must be 280 characters or fewer.\n"
        "- The tone must aggressively reflect the persona.\n"
        "- Do NOT include markdown code fences.\n"
        "- Do NOT deviate from the JSON schema."
    )
    response = llm.invoke([SystemMessage(content=system_prompt)])
    content = response.content.strip()

    # Clean up accidental markdown fences
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: wrap raw text
        parsed = {
            "bot_id": state["bot_id"],
            "topic": topic,
            "post_content": content[:280],
        }

    state["structured_output"] = parsed
    state["post_content"] = parsed.get("post_content", "")
    return state


# ---------------------------------------------------------------------------
# Graph Assembly
# ---------------------------------------------------------------------------
def build_content_engine() -> StateGraph:
    """Assemble and compile the LangGraph state machine."""
    workflow = StateGraph(GraphState)
    workflow.add_node("decide_search", decide_search)
    workflow.add_node("web_search", web_search)
    workflow.add_node("draft_post", draft_post)

    workflow.set_entry_point("decide_search")
    workflow.add_edge("decide_search", "web_search")
    workflow.add_edge("web_search", "draft_post")
    workflow.add_edge("draft_post", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_bot_post(bot_id: str) -> dict:
    """Run the full LangGraph pipeline for a given bot and return the structured JSON output."""
    if bot_id not in BOT_PERSONAS:
        raise ValueError(f"Unknown bot_id: {bot_id}")

    graph = build_content_engine()
    initial_state: GraphState = {
        "bot_id": bot_id,
        "persona": BOT_PERSONAS[bot_id]["description"],
        "topic": "",
        "search_query": "",
        "search_results": "",
        "post_content": "",
        "structured_output": {},
    }
    final_state = graph.invoke(initial_state)
    return final_state["structured_output"]


if __name__ == "__main__":
    print("\n[Phase 2 Test] Running LangGraph for Bot A (Tech Maximalist)...\n")
    result = generate_bot_post("bot_a")
    print("\nFinal structured output:")
    print(json.dumps(result, indent=2))

