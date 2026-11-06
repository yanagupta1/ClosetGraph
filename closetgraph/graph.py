from langgraph.graph import END, START, StateGraph

from closetgraph.candidates import generate_candidates
from closetgraph.models import ClosetGraphState
from closetgraph.research import run_research
from closetgraph.retrieval import retrieve_style_guidance
from closetgraph.stylist import rank_outfits
from closetgraph.supervisor import supervise
from closetgraph.weather import fetch_open_meteo, resolve_weather


def generate_candidates_node(state: ClosetGraphState) -> ClosetGraphState:
    state["candidate_outfits"] = generate_candidates(state["wardrobe"])
    return state


def supervisor_node(state: ClosetGraphState) -> ClosetGraphState:
    """Decide whether this occasion needs external research (hybrid routing)."""
    decision = supervise(state["occasion"])
    state["research_needed"] = decision["research_needed"]
    state["research_query"] = decision["research_query"]
    return state


def research_node(state: ClosetGraphState) -> ClosetGraphState:
    """
    Run the research agent for the supervisor's query. Only reached when research
    is needed. A failure sets research_error and leaves research_context None;
    the run continues (graceful degradation) rather than blocking the ranking.
    """
    context, error = run_research(state["research_query"])
    state["research_context"] = context
    state["research_error"] = error
    return state


def _route_research(state: ClosetGraphState) -> str:
    return "research" if state.get("research_needed") else "skip"


def weather_lookup_node(state: ClosetGraphState) -> ClosetGraphState:
    """
    Resolve the optional weather_input into weather_context (with provenance),
    or None. No weather_input (or no location) means no API call. A failed
    lookup sets weather_error and leaves weather_context None rather than
    fabricating conditions.
    """
    context, error = resolve_weather(
        state.get("weather_input"), fetcher=fetch_open_meteo
    )
    state["weather_context"] = context
    state["weather_error"] = error
    # Deterministic gate: weather is "relevant" when we actually have context.
    # Whether it changes the outfit is the stylist's judgment, not a router's.
    state["weather_relevant"] = context is not None
    return state



def retrieve_style_guidance_node(state: ClosetGraphState) -> ClosetGraphState:
    """
    Real implementation, replaces the empty-list stub.
    Calls retrieve_style_guidance(state["occasion"]), writes result to
    state["retrieved_guidance"].
    """
    state["retrieved_guidance"] = retrieve_style_guidance(
        state["occasion"], weather_context=state.get("weather_context")
    )
    return state


def stylist_node(state: ClosetGraphState) -> ClosetGraphState:
    state["ranked_outfits"] = rank_outfits(
        state["candidate_outfits"],
        state["wardrobe"],
        state["occasion"],
        state["retrieved_guidance"],
        weather_context=state.get("weather_context"),
        research_context=state.get("research_context"),
    )
    return state


def build_graph():
    graph = StateGraph(ClosetGraphState)
    graph.add_node("generate_candidates_node", generate_candidates_node)
    graph.add_node("supervisor_node", supervisor_node)
    graph.add_node("research_node", research_node)
    graph.add_node("weather_lookup_node", weather_lookup_node)
    graph.add_node("retrieve_style_guidance_node", retrieve_style_guidance_node)
    graph.add_node("stylist_node", stylist_node)
    graph.add_edge(START, "generate_candidates_node")
    graph.add_edge("generate_candidates_node", "supervisor_node")
    # Supervisor routes: research when needed, otherwise straight to weather.
    graph.add_conditional_edges(
        "supervisor_node",
        _route_research,
        {"research": "research_node", "skip": "weather_lookup_node"},
    )
    graph.add_edge("research_node", "weather_lookup_node")
    graph.add_edge("weather_lookup_node", "retrieve_style_guidance_node")
    graph.add_edge("retrieve_style_guidance_node", "stylist_node")
    graph.add_edge("stylist_node", END)
    return graph.compile()





