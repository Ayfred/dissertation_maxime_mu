
from huggingface_hub import login
login()

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")
