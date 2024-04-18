from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda"

model_name = "qwen/Qwen1.5-7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Model {model_name} loaded in {device} successfully!")


def response(prompt, max_len=128, remove_prompt=True):
    input_text = prompt

    # Tokenize the input text
    model_inputs = tokenizer.encode_plus(
        input_text, return_tensors="pt").to(device)

    # Response generation
    generated_ids = model.generate(
        input_ids=model_inputs["input_ids"],
        max_new_tokens=max_len,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,  # helps to avoid repetitions
    )

    # Decode the generated response to text
    response_text = tokenizer.decode(
        generated_ids[0], skip_special_tokens=True)

    if (remove_prompt):
        if response_text.startswith(prompt):
            response_text = response_text[len(prompt):].lstrip()
        else:
            print("Remove prompt error: prompt not found in response")

    return response_text


def batch_response(prompts, max_len=128, remove_prompt=True):
    responses = []
    for prompt in prompts:
        responses.append(response(prompt, max_len, remove_prompt))

    return responses
