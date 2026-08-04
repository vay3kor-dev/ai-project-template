import os
import json
import ollama

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# -------------------------
# BMF CONFIG
# -------------------------

api_key = os.getenv("BOSCH_MODEL_FARM_API_KEY")
client = OpenAI(
    api_key="dummy",
    base_url="https://aoai-farm.bosch-temp.com/api/openai/deployments/askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18",
    default_headers={
        "genaiplatform-farm-subscription-key": api_key
    }
)

# -------------------------
# SLM CALL (PHI-3)
# -------------------------

def ask_phi3(question):

    response = ollama.chat(
        model="phi3",
        messages=[
            {
                "role": "system",
                "content": """
You are an AI assistant.

Answer the question and return JSON.

Format:
{
  "answer":"...",
  "confidence":95
}
"""
            },
            {
                "role": "user",
                "content": question
            }
        ],
        format="json"
    )

    return json.loads(
        response["message"]["content"]
    )

# -------------------------
# LLM CALL (BMF)
# -------------------------

def ask_bmf(question):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": question
            }
        ],
        extra_query={
            "api-version": "2024-08-01-preview"
        }
    )

    return response.choices[0].message.content

# -------------------------
# MAIN LOOP
# -------------------------

print("Cascade Chatbot")
print("SLM = Phi-3")
print("LLM = Bosch Model Farm")
print("Type exit to quit\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    try:

        phi_result = ask_phi3(question)

        answer = phi_result["answer"]

        confidence = int(
            phi_result["confidence"]
        )

        print(
            f"\nPhi-3 confidence: {confidence}%"
        )

        if confidence >= 80:

            print("\nUsing SLM (Phi-3)\n")

            print("Bot:", answer)

        else:

            print(
                "\nLow confidence."
            )

            print(
                "Switching to Bosch Model Farm...\n"
            )

            bmf_answer = ask_bmf(question)

            print("Bot:", bmf_answer)

    except Exception as e:

        print(
            "\nPhi-3 failed."
        )

        print(
            "Switching to Bosch Model Farm...\n"
        )

        try:

            bmf_answer = ask_bmf(question)

            print("Bot:", bmf_answer)

        except Exception as llm_error:

            print(
                "Error:",
                llm_error
            )