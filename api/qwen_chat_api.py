from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda"

model_name = "qwen/Qwen1.5-0.5B-Chat"

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2",
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Model {model_name} loaded in {device} successfully!")


def response(prompt, max_len=128):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=max_len
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True)[0]

    return response_text
