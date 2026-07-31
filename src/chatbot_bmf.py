import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("BOSCH_MODEL_FARM_API_KEY")

client = OpenAI(
    api_key="dummy",
    base_url="https://aoai-farm.bosch-temp.com/api/openai/deployments/askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18",
    default_headers={
        "genaiplatform-farm-subscription-key": api_key
    }
)

conversation = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    }
]

print("Bosch Model Farm Chatbot")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    conversation.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation,
        extra_query={
            "api-version": "2024-08-01-preview"
        }
    )

    assistant_reply = response.choices[0].message.content

    print(f"\nBot: {assistant_reply}\n")

    conversation.append(
        {
            "role": "assistant",
            "content": assistant_reply
        }
    )