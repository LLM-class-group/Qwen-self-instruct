from modelscope import AutoModelForCausalLM, AutoTokenizer
import subprocess
import torch

# execute `nvidia-smi`, query the free memory of GPU 0 (MB)
result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader', '--id=0'],
                        capture_output=True, text=True)
free_memory = result.stdout.strip().split('\n')[0]
print(f"GPU 0 free memory: {free_memory} MiB")

cuda_available = int(free_memory) > 5000  # 5GB

device = "cuda" if cuda_available else "cpu"
print(f"Using device: {device}")

model_name = f"qwen/Qwen1.5-1.8B"

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
