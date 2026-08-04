from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import (
    OllamaEmbeddings,
    OllamaLLM
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

DB_DIR = PROJECT_DIR / "chroma_text_db"

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma(
    persist_directory=str(DB_DIR),
    embedding_function=embeddings,
    collection_name="knowledge_base"
)

phi3 = OllamaLLM(
    model="phi3"
)

print("Text RAG Chatbot Ready.")
print("Type /quit to exit.\n")

while True:

    question = input("You: ")

    if question.lower() == "/quit":
        break

    docs = db.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [d.page_content for d in docs]
    )

    prompt = f"""
Use only the provided context.

Context:
{context}

Question:
{question}

If answer not found, say:
"I don't know based on the knowledge base."

Answer:
"""

    response = phi3.invoke(prompt)

    print("\nBot:", response)
    print()