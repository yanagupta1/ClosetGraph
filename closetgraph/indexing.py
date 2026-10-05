"""
closetgraph/indexing.py

Builds the persistent Chroma collection from the style guide corpus.
This is a ONE-TIME build step, not something that runs on every graph
invocation, the retrieval node queries an already-built collection,
it does not rebuild it.
"""

import re

import chromadb

CORPUS_FILES = {
    "occasion": "style_guide/occasion.md",
    "layering": "style_guide/layering.md",
    "garments": "style_guide/garments.md",
    "color": "style_guide/color.md",
}
# weather.md intentionally excluded, not indexed until Stage 3

def chunk_markdown_file(filepath: str, category: str) -> list[dict]:
    """
    Split a corpus file on '## ' headings. One chunk per section.
    Each chunk dict: {id, text, category, heading}.
    - id: slug of category + heading, e.g. "layering__structured-layers"
    - text: the full section including its heading line, so the
      heading's own words are part of what gets embedded
    - category: from CORPUS_FILES mapping
    - heading: the raw heading text, kept as metadata for debugging/traces
    """
    with open(filepath, encoding="utf-8") as handle:
        content = handle.read()

    chunks: list[dict] = []
    heading: str | None = None
    lines: list[str] = []

    def flush() -> None:
        if heading is not None:
            slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
            chunks.append(
                {
                    "id": f"{category}__{slug}",
                    "text": "\n".join(lines).rstrip(),
                    "category": category,
                    "heading": heading,
                }
            )

    for line in content.splitlines():
        if line.startswith("## "):
            flush()
            heading = line[3:].strip()
            lines = [line]
        elif heading is not None:
            lines.append(line)
    flush()

    return chunks


def build_style_guide_index(persist_path: str = "./chroma_db") -> None:
    """
    1. chromadb.PersistentClient(path=persist_path)
    2. get_or_create_collection("style_guide")
       (uses Chroma's default local embedding function, nothing else
       configured)
    3. For each (category, filepath) in CORPUS_FILES, chunk the file,
       add all chunks: collection.add(documents=[text,...],
       metadatas=[{"category":, "heading":}, ...], ids=[id,...])
    4. If the collection already has entries, clear and rebuild rather
       than appending duplicates (this is a rebuild-from-scratch
       script, run manually whenever corpus files change).
    5. Print a short summary: how many chunks per category.
    """
    client = chromadb.PersistentClient(path=persist_path)

    if "style_guide" in [c.name for c in client.list_collections()]:
        if client.get_collection("style_guide").count() > 0:
            client.delete_collection("style_guide")

    collection = client.get_or_create_collection("style_guide")

    for category, filepath in CORPUS_FILES.items():
        chunks = chunk_markdown_file(filepath, category)
        collection.add(
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[
                {"category": chunk["category"], "heading": chunk["heading"]}
                for chunk in chunks
            ],
            ids=[chunk["id"] for chunk in chunks],
        )
        print(f"{category}: {len(chunks)} chunks")


if __name__ == "__main__":
    build_style_guide_index()

