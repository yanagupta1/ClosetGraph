from pydantic import BaseModel
from typing import Literal, TypedDict


class WardrobeItem(BaseModel):
    id: str
    name: str  # free text, e.g. "black scoop-neck tank top"
    category: Literal["top", "bottom", "layering"]


class OutfitCandidate(BaseModel):
    items: list[str]  # WardrobeItem ids


class RankedOutfit(BaseModel):
    items: list[str]
    rank: int
    reasoning: str


class ClosetGraphState(TypedDict):
    wardrobe: list[WardrobeItem]
    occasion: str
    candidate_outfits: list[OutfitCandidate]
    ranked_outfits: list[RankedOutfit]
