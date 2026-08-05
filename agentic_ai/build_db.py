from pathlib import Path
import shutil

import chromadb
import ollama


BASE_DIR = Path(__file__).resolve().parent

KNOWLEDGE_FILE = BASE_DIR / "knowledge.txt"
CHROMA_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "agentic_kb"
EMBEDDING_MODEL = "nomic-embed-text"


def read_knowledge_file(file_path: Path) -> str:
    """
    Reads the knowledge.txt file.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Knowledge file was not found: {file_path}"
        )

    return file_path.read_text(
        encoding="utf-8"
    )


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 80
) -> list:
    """
    Splits text into overlapping chunks.
    """

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def create_embeddings(chunks: list[str]) -> list[list[float]]:
    """
    Creates embeddings using the local Ollama
    nomic-embed-text model.
    """

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=chunks
    )

    return response["embeddings"]


def build_database() -> None:
    """
    Builds a new ChromaDB knowledge base.
    """

    print("Reading knowledge file...")

    text = read_knowledge_file(
        KNOWLEDGE_FILE
    )

    print("Splitting text into chunks...")

    chunks = chunk_text(text)

    if not chunks:
        raise ValueError(
            "knowledge.txt is empty."
        )

    print(
        f"Total chunks created: {len(chunks)}"
    )

    print(
        "Creating embeddings with "
        f"{EMBEDDING_MODEL}..."
    )

    embeddings = create_embeddings(chunks)

    # Delete the old database completely before rebuilding.
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)

    print("Creating ChromaDB collection...")

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    ids = [
        f"chunk_{index}"
        for index in range(len(chunks))
    ]

    metadatas = [
        {
            "source": KNOWLEDGE_FILE.name,
            "chunk_id": index
        }
        for index in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("Database created successfully.")
    print(f"ChromaDB folder: {CHROMA_PATH}")
    print(f"Collection name: {COLLECTION_NAME}")


if __name__ == "__main__":
    build_database()