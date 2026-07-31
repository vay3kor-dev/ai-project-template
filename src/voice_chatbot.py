import requests
import pyttsx3

try:
    import speech_recognition as sr
    VOICE_INPUT_AVAILABLE = True
except ImportError:
    VOICE_INPUT_AVAILABLE = False


MODEL_NAME = "qwen3"
OLLAMA_URL = "http://localhost:11434/api/chat"


# Text-to-speech setup
engine = pyttsx3.init()
engine.setProperty("rate", 165)


def speak(text):
    print("\nQwen3:", text, "\n")
    engine.say(text)
    engine.runAndWait()


def ask_ollama(messages):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False
        }
    )

    data = response.json()
    return data["message"]["content"]


def listen_from_microphone():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening... speak now.")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except Exception as e:
        print("Could not understand voice. Error:", e)
        return ""


def main():
    print("Voice Chatbot Started")
    print("Type or say 'exit' to stop.\n")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Answer clearly and simply."
        }
    ]

    while True:
        if VOICE_INPUT_AVAILABLE:
            choice = input("Press Enter to speak, or type your message: ")

            if choice.strip():
                user_input = choice
            else:
                user_input = listen_from_microphone()
        else:
            user_input = input("You: ")

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "bye"]:
            speak("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        answer = ask_ollama(messages)

        messages.append({"role": "assistant", "content": answer})

        speak(answer)


if __name__ == "__main__":
    main()