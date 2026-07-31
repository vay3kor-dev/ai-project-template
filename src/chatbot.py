import requests

print("Qwen3 Chatbot Started!")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen3",
            "prompt": user_input,
            "stream": False
        }
    )

    answer = response.json()["response"]

    print(f"\nQwen3: {answer}\n")