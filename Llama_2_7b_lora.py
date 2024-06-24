import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# Load a small subset of the IMDb dataset
def load_small_imdb():
    dataset = load_dataset("imdb")
    small_train_dataset = dataset["train"].shuffle(seed=42).select(range(100))  # Select 100 examples for training
    small_test_dataset = dataset["test"].shuffle(seed=42).select(range(20))    # Select 20 examples for testing
    return {"train": small_train_dataset, "test": small_test_dataset}

dataset = load_small_imdb()

# Specify the path to your local LLaMA 2 7B model
model_path = "/Users/charaka/Documents/charaka/llama2-7b"

# Load tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_path)

# Ensure the padding token is set
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Load model
model = LlamaForCausalLM.from_pretrained(model_path)

# Resize token embeddings in the model if needed
model.resize_token_embeddings(len(tokenizer))

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # The dimension of the low-rank adaptation
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none"
)

# Apply the LoRA configuration to the model
model = get_peft_model(model, lora_config)

# Tokenize dataset
def tokenize_function(examples):
    outputs = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
    outputs['labels'] = outputs['input_ids'].copy()  # Copy input_ids to labels for language modeling
    return outputs

tokenized_datasets = dataset["train"].map(tokenize_function, batched=True, remove_columns=["label"])
tokenized_test_dataset = dataset["test"].map(tokenize_function, batched=True, remove_columns=["label"])

# Determine the device to use
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Masked Language Modeling is not used here
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_test_dataset,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Save the model
trainer.save_model("./fine-tuned-llama-lora")
tokenizer.save_pretrained("./fine-tuned-llama-lora")

# Load the fine-tuned model and tokenizer for inference
model = LlamaForCausalLM.from_pretrained("./fine-tuned-llama-lora")
tokenizer = LlamaTokenizer.from_pretrained("./fine-tuned-llama-lora")

# Inference function
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the model with a sample input
prompt = "It was a terrible movie because"
generated_text = generate_text(prompt)
print(f"Prompt: {prompt}\nGenerated Text: {generated_text}")