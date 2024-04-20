from openai import OpenAI
from os import getenv
import json

# gets API Key from environment variable
client = OpenAI(
    base_url="https://lonlie.plus7.plus/v1",
    api_key=getenv("OPENAI_API_KEY"),
)

log_path = "/home/jiahe/LLMs/lima/qwen_self_instruct/api/log/openai.jsonl"


def response(prompt, max_len=128):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
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
        print(f"Warning: Could not write openai api response to log file: {e}")

    return response
