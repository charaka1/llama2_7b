import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

# Download the Llama 2 7B model from Hugging Face using your access token
access_token = "Your_Hugging_Face_Access_Token"
model_name = "meta-llama/Llama-2-7b-chat-hf"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)

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
