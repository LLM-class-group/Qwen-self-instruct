from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda"

model_name = "qwen/Qwen1.5-7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2",
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Model {model_name} loaded in {device} successfully!")


def response(prompt, max_len=128):
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

    # Remove the input text from the generated response
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the generated response to text
    response_text = tokenizer.decode(
        generated_ids[0], skip_special_tokens=True)

    return response_text


def batch_response(prompts, max_len=128):
    responses = []
    for prompt in prompts:
        responses.append(response(prompt, max_len))

    return responses
