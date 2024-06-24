import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

# Specify the local directory where your LLaMA 2 7B model is saved
model_directory = "/Users/charaka/Documents/charaka/llama2-7b"

# Load tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_directory)

# Load model
model = LlamaForCausalLM.from_pretrained(model_directory, torch_dtype=torch.float16)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Inference function
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the model with a sample input
prompt = "The movie was terrible because"
generated_text = generate_text(prompt)
print(f"Prompt: {prompt}\nGenerated Text: {generated_text}")
