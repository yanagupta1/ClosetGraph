from langgraph.graph import END, START, StateGraph

from closetgraph.candidates import generate_candidates
from closetgraph.models import ClosetGraphState
from closetgraph.retrieval import retrieve_style_guidance
from closetgraph.stylist import rank_outfits
from closetgraph.weather import fetch_open_meteo, resolve_weather


def generate_candidates_node(state: ClosetGraphState) -> ClosetGraphState:
    state["candidate_outfits"] = generate_candidates(state["wardrobe"])
    return state


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
    return state



def retrieve_style_guidance_node(state: ClosetGraphState) -> ClosetGraphState:
    """
    Real implementation, replaces the empty-list stub.
    Calls retrieve_style_guidance(state["occasion"]), writes result to
    state["retrieved_guidance"].
    """
    state["retrieved_guidance"] = retrieve_style_guidance(state["occasion"])
    return state


def stylist_node(state: ClosetGraphState) -> ClosetGraphState:
    state["ranked_outfits"] = rank_outfits(
        state["candidate_outfits"],
        state["wardrobe"],
        state["occasion"],
        state["retrieved_guidance"],
    )
    return state


def build_graph():
    graph = StateGraph(ClosetGraphState)
    graph.add_node("generate_candidates_node", generate_candidates_node)
    graph.add_node("weather_lookup_node", weather_lookup_node)
    graph.add_node("retrieve_style_guidance_node", retrieve_style_guidance_node)
    graph.add_node("stylist_node", stylist_node)
    graph.add_edge(START, "generate_candidates_node")
    graph.add_edge("generate_candidates_node", "weather_lookup_node")
    graph.add_edge("weather_lookup_node", "retrieve_style_guidance_node")
    graph.add_edge("retrieve_style_guidance_node", "stylist_node")
    graph.add_edge("stylist_node", END)
    return graph.compile()





