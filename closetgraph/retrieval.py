"""
closetgraph/retrieval.py
"""

import chromadb

CATEGORIES = ["occasion", "layering", "garments", "color"]

def retrieve_style_guidance(
    occasion: str, persist_path: str = "./chroma_db", k_per_category: int = 2
) -> list[str]:
    """
    Stratified retrieval, NOT a single top-k over the whole collection.
    1. Load the persistent collection (read-only, do not rebuild here).
    2. For each category in CATEGORIES, run collection.query(
         query_texts=[occasion], n_results=k_per_category,
         where={"category": category})
    3. Concatenate all results' documents into one flat list.
       Max 8 chunks total (4 categories x 2). Preserve category order
       (occasion, layering, garments, color) rather than re-sorting by
       score, so output is deterministic and easy to debug in traces.
    4. Return list[str] of chunk texts.
    """
    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_collection("style_guide")

    guidance: list[str] = []
    for category in CATEGORIES:
        result = collection.query(
            query_texts=[occasion],
            n_results=k_per_category,
            where={"category": category},
        )
        documents = (result.get("documents") or [[]])[0]
        guidance.extend(documents)

    return guidance[:8]

