from pathlib import Path
import ollama
import whisper

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma


# -----------------------------
# Paths
# -----------------------------
PDF_PATH = "docs/sample.pdf"
IMAGE_PATH = "inputs/image.jpg"
AUDIO_PATH = "inputs/audio.wav"
CHROMA_DIR = "chroma_db"


# -----------------------------
# 1. Image to text using LLaVA
# -----------------------------
def image_to_text(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        return "No image file found."

    response = ollama.chat(
        model="llava",
        messages=[
            {
                "role": "user",
                "content": (
                    "Describe this image clearly. "
                    "If it contains text, extract the visible text also."
                ),
                "images": [str(image_path)]
            }
        ]
    )

    return response["message"]["content"]


# -----------------------------
# 2. Audio to text using Whisper
# -----------------------------
def audio_to_text(audio_path):
    audio_path = Path(audio_path)

    if not audio_path.exists():
        return "No audio file found."

    model = whisper.load_model("base")
    result = model.transcribe(str(audio_path))

    return result["text"]


# -----------------------------
# 3. Create PDF vector database
# -----------------------------
def build_pdf_vector_db(pdf_path):
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        print("No PDF found. Skipping PDF RAG.")
        return None

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_db


# -----------------------------
# 4. Retrieve relevant PDF text
# -----------------------------
def retrieve_pdf_context(vector_db, query):
    if vector_db is None:
        return "No PDF context available."

    results = vector_db.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in results])

    return context


# -----------------------------
# 5. Generate final summary
# -----------------------------
def generate_final_summary(image_text, audio_text, pdf_context):
    llm = OllamaLLM(model="qwen3")

    prompt = f"""
You are an AI assistant.

You are given three types of input:

1. IMAGE DESCRIPTION:
{image_text}

2. AUDIO TRANSCRIPT:
{audio_text}

3. PDF CONTEXT:
{pdf_context}

Task:
Create a clear summary using all available information.

Output format:
- Overall Summary
- Important Points
- Action Items, if any
- References from PDF, if useful
"""

    response = llm.invoke(prompt)
    return response


# -----------------------------
# Main flow
# -----------------------------
if __name__ == "__main__":
    print("Step 1: Converting image to text...")
    image_text = image_to_text(IMAGE_PATH)
    print("\nIMAGE TEXT:\n", image_text)

    print("\nStep 2: Converting audio to text...")
    audio_text = audio_to_text(AUDIO_PATH)
    print("\nAUDIO TEXT:\n", audio_text)

    print("\nStep 3: Building PDF vector database...")
    vector_db = build_pdf_vector_db(PDF_PATH)

    print("\nStep 4: Retrieving PDF context...")
    combined_query = image_text + "\n" + audio_text
    pdf_context = retrieve_pdf_context(vector_db, combined_query)
    print("\nPDF CONTEXT:\n", pdf_context[:1000])

    print("\nStep 5: Generating final summary...")
    final_summary = generate_final_summary(image_text, audio_text, pdf_context)

    print("\nFINAL SUMMARY:\n")
    print(final_summary)