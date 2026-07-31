from pathlib import Path
import hashlib
import os

import chromadb
import ollama
import whisper
from pypdf import PdfReader


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"

CHAT_MODEL = "qwen3"
VISION_MODEL = "llava"
EMBEDDING_MODEL = "nomic-embed-text"

COLLECTION_NAME = "multimodal_knowledge"


# =========================================================
# OPTIONAL VOICE LIBRARIES
# =========================================================

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

try:
    import pyttsx3
    TEXT_TO_SPEECH_AVAILABLE = True
except ImportError:
    TEXT_TO_SPEECH_AVAILABLE = False


# =========================================================
# GLOBAL CHATBOT MEMORY
# =========================================================

conversation_history = []

latest_image_description = ""
latest_audio_transcript = ""

whisper_model = None
chroma_collection = None


# =========================================================
# CHROMADB SETUP
# =========================================================

def initialize_chroma():
    """
    Creates or opens a persistent ChromaDB collection.
    """

    global chroma_collection

    # If chroma_db accidentally exists as a file, stop with a clear message.
    if CHROMA_DIR.exists() and CHROMA_DIR.is_file():
        raise RuntimeError(
            f"{CHROMA_DIR} is a file, but it must be a folder. "
            "Delete that file and run the program again."
        )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    chroma_collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    print(f"Vector database ready: {CHROMA_DIR}")


# =========================================================
# OLLAMA EMBEDDINGS
# =========================================================

def create_embedding(text):
    """
    Converts text into a numerical embedding using Ollama.
    """

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text
    )

    # Supports dictionary-style and object-style Ollama responses.
    if isinstance(response, dict):
        return response["embeddings"][0]

    return response.embeddings[0]


# =========================================================
# TEXT CHUNKING
# =========================================================

def split_text(text, chunk_size=1000, overlap=150):
    """
    Splits long text into smaller overlapping chunks.
    """

    text = text.strip()

    if not text:
        return []

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


# =========================================================
# ADD CONTENT TO CHROMADB
# =========================================================

def store_chunks(chunks, source_name, source_type):
    """
    Stores text chunks and embeddings in ChromaDB.
    """

    if not chunks:
        print("No readable text was found.")
        return

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for index, chunk in enumerate(chunks):
        unique_text = f"{source_name}-{index}-{chunk}"
        chunk_id = hashlib.md5(unique_text.encode("utf-8")).hexdigest()

        embedding = create_embedding(chunk)

        ids.append(chunk_id)
        documents.append(chunk)
        embeddings.append(embedding)
        metadatas.append(
            {
                "source": source_name,
                "type": source_type,
                "chunk_number": index
            }
        )

    # upsert avoids duplicate-ID errors when the same file is added again
    chroma_collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        f"Stored {len(chunks)} chunk(s) from "
        f"{source_name} in ChromaDB."
    )


# =========================================================
# IMAGE PROCESSING USING LLAVA
# =========================================================

def process_image(image_path):
    """
    Sends an image to LLaVA and returns a detailed description.
    """

    global latest_image_description

    path = Path(image_path).expanduser()

    if not path.is_absolute():
        path = PROJECT_DIR / path

    path = path.resolve()

    if not path.exists():
        return f"Image file not found: {path}"

    supported_extensions = {
        ".jpg", ".jpeg", ".png", ".webp", ".bmp"
    }

    if path.suffix.lower() not in supported_extensions:
        return (
            "Unsupported image format. Use JPG, JPEG, PNG, "
            "WEBP, or BMP."
        )

    print("Analyzing image with LLaVA...")

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Analyze this image carefully. "
                    "Describe the main objects, background, colors, "
                    "visible text, important details, and likely purpose. "
                    "Do not invent details that are not visible."
                ),
                "images": [str(path)]
            }
        ]
    )

    latest_image_description = response["message"]["content"]

    store_chunks(
        chunks=[latest_image_description],
        source_name=path.name,
        source_type="image"
    )

    return latest_image_description


# =========================================================
# AUDIO PROCESSING USING WHISPER
# =========================================================

def load_whisper_model():
    """
    Loads Whisper only when audio processing is first requested.
    """

    global whisper_model

    if whisper_model is None:
        print("Loading Whisper model...")
        whisper_model = whisper.load_model("base")

    return whisper_model


def process_audio(audio_path):
    """
    Converts an audio file to text using Whisper.
    """

    global latest_audio_transcript

    path = Path(audio_path).expanduser()

    if not path.is_absolute():
        path = PROJECT_DIR / path

    path = path.resolve()

    if not path.exists():
        return f"Audio file not found: {path}"

    supported_extensions = {
        ".wav", ".mp3", ".m4a", ".flac", ".ogg"
    }

    if path.suffix.lower() not in supported_extensions:
        return (
            "Unsupported audio format. Use WAV, MP3, M4A, "
            "FLAC, or OGG."
        )

    model = load_whisper_model()

    print("Transcribing audio with Whisper...")

    result = model.transcribe(str(path))
    latest_audio_transcript = result["text"].strip()

    if not latest_audio_transcript:
        return "Whisper could not detect understandable speech."

    chunks = split_text(latest_audio_transcript)

    store_chunks(
        chunks=chunks,
        source_name=path.name,
        source_type="audio"
    )

    return latest_audio_transcript


# =========================================================
# PDF PROCESSING
# =========================================================

def process_pdf(pdf_path):
    """
    Extracts text from a PDF and stores it in ChromaDB.
    """

    path = Path(pdf_path).expanduser()

    if not path.is_absolute():
        path = PROJECT_DIR / path

    path = path.resolve()

    if not path.exists():
        return f"PDF file not found: {path}"

    if path.suffix.lower() != ".pdf":
        return "The selected file is not a PDF."

    print("Reading PDF...")

    try:
        reader = PdfReader(str(path))
    except Exception as error:
        return (
            "The PDF could not be opened. Make sure it is a real PDF "
            f"and not a text file renamed to .pdf.\nDetails: {error}"
        )

    pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text and page_text.strip():
            pages_text.append(
                f"Page {page_number}\n{page_text.strip()}"
            )

    complete_text = "\n\n".join(pages_text)

    if not complete_text:
        return (
            "No readable text was extracted. The PDF may contain "
            "only scanned images and may require OCR."
        )

    chunks = split_text(complete_text)

    store_chunks(
        chunks=chunks,
        source_name=path.name,
        source_type="pdf"
    )

    return (
        f"PDF '{path.name}' was added successfully. "
        f"{len(reader.pages)} page(s) were checked and "
        f"{len(chunks)} text chunk(s) were stored."
    )


# =========================================================
# RETRIEVE RELEVANT KNOWLEDGE
# =========================================================

def retrieve_context(question, number_of_results=5):
    """
    Searches ChromaDB for content relevant to the user's question.
    """

    if chroma_collection.count() == 0:
        return "", []

    question_embedding = create_embedding(question)

    result = chroma_collection.query(
        query_embeddings=[question_embedding],
        n_results=min(number_of_results, chroma_collection.count())
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    context_sections = []
    sources = []

    for document, metadata in zip(documents, metadatas):
        source = metadata.get("source", "Unknown source")
        source_type = metadata.get("type", "unknown")

        context_sections.append(
            f"Source: {source} ({source_type})\n{document}"
        )

        if source not in sources:
            sources.append(source)

    return "\n\n".join(context_sections), sources


# =========================================================
# CHAT WITH QWEN
# =========================================================

def ask_chatbot(user_question):
    """
    Answers using retrieved multimodal content and chat history.
    """

    context, sources = retrieve_context(user_question)

    system_message = """
You are a helpful local multimodal chatbot.

You can answer ordinary questions and questions about content extracted
from images, audio files, and PDFs.

Rules:
1. Use the supplied retrieved context when it is relevant.
2. Do not invent information that is not in the supplied context.
3. If the answer is not present in the supplied content, clearly say so.
4. When the question is about an uploaded file, mention the source file.
5. Give clear and easy-to-understand answers.
6. Maintain the conversation naturally.
"""

    if context:
        current_prompt = f"""
Retrieved content from the user's files:

{context}

User question:

{user_question}
"""
    else:
        current_prompt = user_question

    messages = [
        {
            "role": "system",
            "content": system_message
        }
    ]

    # Keep only recent messages to avoid an excessively long prompt.
    messages.extend(conversation_history[-10:])

    messages.append(
        {
            "role": "user",
            "content": current_prompt
        }
    )

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=messages
    )

    answer = response["message"]["content"]

    conversation_history.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    conversation_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return answer, sources


# =========================================================
# COMBINED SUMMARY
# =========================================================

def create_combined_summary():
    """
    Creates a summary from all content currently stored in ChromaDB.
    """

    if chroma_collection.count() == 0:
        return (
            "No image, audio, or PDF content has been added yet. "
            "Use /image, /audio, or /pdf first."
        )

    result = chroma_collection.get(
        include=["documents", "metadatas"]
    )

    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    all_content = []

    for document, metadata in zip(documents, metadatas):
        source = metadata.get("source", "Unknown source")
        source_type = metadata.get("type", "unknown")

        all_content.append(
            f"Source: {source} ({source_type})\n{document}"
        )

    combined_content = "\n\n".join(all_content)

    # Prevent an extremely large prompt in a prototype.
    combined_content = combined_content[:20000]

    prompt = f"""
Create a combined summary of the following multimodal information.

Separate the response into:
1. Overall Summary
2. Image Information
3. Audio Information
4. PDF Information
5. Important Points
6. Action Items, if any
7. Sources Used

Content:

{combined_content}
"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# =========================================================
# MICROPHONE INPUT
# =========================================================

def listen_from_microphone():
    """
    Records one question from the microphone.
    """

    if not SPEECH_RECOGNITION_AVAILABLE:
        return (
            None,
            "SpeechRecognition is not installed. "
            "Run: pip install SpeechRecognition"
        )

    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source)

        print("Converting speech to text...")

        text = recognizer.recognize_google(audio)
        return text, None

    except Exception as error:
        return None, f"Microphone input failed: {error}"


# =========================================================
# TEXT-TO-SPEECH
# =========================================================

def speak_text(text):
    """
    Reads the chatbot response aloud.
    """

    if not TEXT_TO_SPEECH_AVAILABLE:
        print(
            "Text-to-speech is unavailable. "
            "Install it using: pip install pyttsx3"
        )
        return

    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


# =========================================================
# DISPLAY HELP
# =========================================================

def show_help():
    print(
        """
Available commands:

/image <path>
    Analyze an image and add its description to chatbot memory.
    Example:
    /image inputs/test_image.jpg

/audio <path>
    Transcribe an audio file and add its transcript to memory.
    Example:
    /audio inputs/test_audio.wav

/pdf <path>
    Read a PDF and add its content to the RAG database.
    Example:
    /pdf docs/sample.pdf

/voice
    Ask a question using the microphone.

/summary
    Create a combined summary of all added content.

/sources
    Show how many entries are stored in the vector database.

/clear-chat
    Clear only the current conversation history.

/speak-on
    Turn spoken chatbot answers on.

/speak-off
    Turn spoken chatbot answers off.

/help
    Display this command list.

/quit
    Close the chatbot.

You can also type a normal question without using a command.
"""
    )


# =========================================================
# MAIN CHATBOT LOOP
# =========================================================

def run_chatbot():
    initialize_chroma()

    speak_answers = False

    print("\nLocal Multimodal RAG Chatbot")
    print("Type /help to see commands.")
    print("Type /quit to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nChatbot closed.")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print("Chatbot: Goodbye!")
            break

        elif user_input.lower() == "/help":
            show_help()
            continue

        elif user_input.lower().startswith("/image "):
            image_path = user_input[7:].strip()
            result = process_image(image_path)

            print("\nChatbot: Image processed.")
            print(result)
            print()
            continue

        elif user_input.lower().startswith("/audio "):
            audio_path = user_input[7:].strip()
            result = process_audio(audio_path)

            print("\nChatbot: Audio processing result:")
            print(result)
            print()
            continue

        elif user_input.lower().startswith("/pdf "):
            pdf_path = user_input[5:].strip()
            result = process_pdf(pdf_path)

            print(f"\nChatbot: {result}\n")
            continue

        elif user_input.lower() == "/summary":
            summary = create_combined_summary()

            print(f"\nChatbot:\n{summary}\n")

            if speak_answers:
                speak_text(summary)

            continue

        elif user_input.lower() == "/sources":
            count = chroma_collection.count()

            print(
                f"\nChatbot: The vector database currently contains "
                f"{count} stored content chunk(s).\n"
            )
            continue

        elif user_input.lower() == "/clear-chat":
            conversation_history.clear()

            print(
                "\nChatbot: Conversation history cleared. "
                "Uploaded knowledge remains available.\n"
            )
            continue

        elif user_input.lower() == "/speak-on":
            speak_answers = True
            print("\nChatbot: Spoken answers enabled.\n")
            continue

        elif user_input.lower() == "/speak-off":
            speak_answers = False
            print("\nChatbot: Spoken answers disabled.\n")
            continue

        elif user_input.lower() == "/voice":
            spoken_text, error = listen_from_microphone()

            if error:
                print(f"\nChatbot: {error}\n")
                continue

            print(f"\nYou said: {spoken_text}")

            answer, sources = ask_chatbot(spoken_text)

        else:
            answer, sources = ask_chatbot(user_input)

        print(f"\nChatbot: {answer}")

        if sources:
            print("\nSources used:")
            for source in sources:
                print(f"- {source}")

        print()

        if speak_answers:
            speak_text(answer)


if __name__ == "__main__":
    run_chatbot()