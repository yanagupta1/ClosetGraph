from langgraph.graph import END, START, StateGraph

from closetgraph.candidates import generate_candidates
from closetgraph.models import ClosetGraphState
from closetgraph.retrieval import retrieve_style_guidance
from closetgraph.stylist import rank_outfits


def generate_candidates_node(state: ClosetGraphState) -> ClosetGraphState:
    state["candidate_outfits"] = generate_candidates(state["wardrobe"])
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
    graph.add_node("retrieve_style_guidance_node", retrieve_style_guidance_node)
    graph.add_node("stylist_node", stylist_node)
    graph.add_edge(START, "generate_candidates_node")
    graph.add_edge("generate_candidates_node", "retrieve_style_guidance_node")
    graph.add_edge("retrieve_style_guidance_node", "stylist_node")
    graph.add_edge("stylist_node", END)
    return graph.compile()





