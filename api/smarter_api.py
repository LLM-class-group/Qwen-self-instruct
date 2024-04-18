from openai import OpenAI
from os import getenv

# gets API Key from environment variable OPENAI_API_KEY
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv(""),
)

completion = client.chat.completions.create(
    model="openai/gpt-3.5-turbo",
    messages=[
        {
          "role": "user",
          "content": "You are a helpful assistant.",
        },
    ],
)

print(completion.choices[0].message.content)
