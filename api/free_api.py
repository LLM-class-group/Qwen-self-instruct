from openai import OpenAI
from os import getenv
import json

# gets API Key from environment variable
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)

log_path = "/home/jiahe/LLMs/lima/qwen_self_instruct/api/log/free.jsonl"

# some free models on OpenRouter
models = [
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-7b-it:free",
    "openchat/openchat-7b:free",
    "nousresearch/nous-capybara-7b:free",
    "undi95/toppy-m-7b:free",
]


def response(prompt, max_len=128, model_id=0):
    completion = client.chat.completions.create(
        model=models[model_id],
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=max_len,
    )

    response = completion.choices[0].message.content

    log_entry = {
        "prompt": prompt,
        "response": response
    }

    try:
        with open(log_path, 'a') as log_file:
            json.dump(log_entry, log_file)
            log_file.write("\n")  # Add newline to separate entries
    except Exception as e:
        print(f"Warning: Could not write free api response to log file: {e}")

    return response
