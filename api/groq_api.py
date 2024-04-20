# Note: Groq api isn't available yet for network reason 

import json

from groq import Groq
from os import getenv

client = Groq(
    api_key=getenv("GROQ_API_KEY"),
)

log_path = "/home/jiahe/LLMs/lima/qwen_self_instruct/api/log/groq.jsonl"

# some free models on Groq
models = [
    "llama3-70b-8192",
    "llama2-70b-4096",
    "mixtral-8x7b-32768",
    "llama3-8b-8192",
    "gemma-7b-it",
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
        print(f"Warning: Could not write groq api response to log file: {e}")

    return response
