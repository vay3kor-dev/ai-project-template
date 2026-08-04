from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


PROJECT_DIR = Path(__file__).resolve().parent.parent

TEXT_FILE = PROJECT_DIR / "data" / "knowledge.txt"

DB_DIR = PROJECT_DIR / "chroma_text_db"

COLLECTION_NAME = "knowledge_base"


print("Reading text file...")

with open(TEXT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

documents = [
    Document(page_content=chunk)
    for chunk in chunks
]

print(f"Chunks created: {len(documents)}")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Building ChromaDB...")

Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=str(DB_DIR),
    collection_name=COLLECTION_NAME
)

print("Database Created Successfully.")