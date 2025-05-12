import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "tiiuae/falcon-rw-1b"  # Fast, open-access

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="auto"
)

def generate_response(prompt: str) -> dict:
    start_time = time.time()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.8,
            top_p=0.95
        )
    duration = round(time.time() - start_time) 
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_tokens = outputs[0].shape[0] - input_tokens

    return {
        "prompt": prompt,
        "response": output_text.strip(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency": duration
    }
