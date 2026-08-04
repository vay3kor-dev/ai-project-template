from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = PROJECT_DIR / "docs" / "sample.pdf"
CHROMA_DIR = PROJECT_DIR / "chroma_phi3_db"

CHAT_MODEL = "phi3"
EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "phi3_pdf_documents"


# ---------------------------------------------------------
# Load and split the PDF
# ---------------------------------------------------------

def load_and_split_pdf(pdf_path):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Loading PDF: {pdf_path.name}")

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    print(f"Loaded {len(documents)} page(s).")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} text chunk(s).")

    return chunks


# ---------------------------------------------------------
# Create or load ChromaDB
# ---------------------------------------------------------

def create_vector_database(chunks):
    print("Creating embeddings using nomic-embed-text...")

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    vector_database = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME
    )

    print(f"Vector database created at: {CHROMA_DIR}")

    return vector_database


def load_existing_vector_database():
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )


# ---------------------------------------------------------
# Prepare the vector database
# ---------------------------------------------------------

def prepare_vector_database():
    database_file = CHROMA_DIR / "chroma.sqlite3"

    if database_file.exists():
        print("Loading the existing ChromaDB database...")
        return load_existing_vector_database()

    print("No existing database found.")
    print("Processing the PDF for the first time...")

    chunks = load_and_split_pdf(PDF_PATH)

    return create_vector_database(chunks)


# ---------------------------------------------------------
# Search for relevant PDF chunks
# ---------------------------------------------------------

def retrieve_context(vector_database, question):
    relevant_documents = vector_database.similarity_search(
        question,
        k=3
    )

    context_parts = []

    for document in relevant_documents:
        page_number = document.metadata.get("page", 0) + 1

        context_parts.append(
            f"PDF page {page_number}:\n"
            f"{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    return context, relevant_documents


# ---------------------------------------------------------
# Ask Phi-3 using retrieved PDF content
# ---------------------------------------------------------

def ask_phi3(question, context):
    phi3 = OllamaLLM(
        model=CHAT_MODEL,
        temperature=0
    )

    prompt = f"""
You are a PDF question-answering assistant.

Answer the user's question using only the PDF context supplied below.

Rules:
1. Do not use unsupported information.
2. If the context does not contain the answer, say:
   "I could not find that information in the PDF."
3. Give a clear and concise answer.
4. Do not mention facts that are not present in the context.

PDF CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    return phi3.invoke(prompt)


# ---------------------------------------------------------
# Continuous chatbot
# ---------------------------------------------------------

def run_chatbot():
    print("\nStarting Phi-3 PDF RAG Chatbot...\n")

    vector_database = prepare_vector_database()

    print("\nChatbot is ready.")
    print("Ask questions about sample.pdf.")
    print("Type /quit to stop.\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() == "/quit":
            print("Chatbot: Goodbye!")
            break

        print("Searching the PDF...")

        context, documents = retrieve_context(
            vector_database,
            question
        )

        answer = ask_phi3(
            question,
            context
        )

        print(f"\nChatbot: {answer}\n")

        print("Retrieved PDF pages:")

        shown_pages = set()

        for document in documents:
            page_number = document.metadata.get("page", 0) + 1

            if page_number not in shown_pages:
                print(f"- Page {page_number}")
                shown_pages.add(page_number)

        print()


if __name__ == "__main__":
    run_chatbot()