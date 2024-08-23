# Use the official PyTorch CUDA Linux image as the base with specified platform
FROM --platform=linux/amd64 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Install HuggingFace Transformers library 
RUN pip install torch transformers datasets accelerate peft sentencepiece

# Set working directory
WORKDIR /app

# Copy your Python source file and data file into the container
# COPY Llama_2_7b_inference.py Llama_2_7b_lora.py /app/

# Define entry point
#CMD ["python", "Llama_2_7b_inference.py"]

# Define entry point
#CMD ["bash"]

