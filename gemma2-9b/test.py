from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("/home/mmu/spinning-storage/mmu/gemma2/gemma-2-9b-it/")
model = AutoModelForCausalLM.from_pretrained(
    "/home/mmu/spinning-storage/mmu/gemma2/gemma-2-9b-it/",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
max_length = 5000 

input_text = "Generate fictional patients dataset."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = outputs = model.generate(input_ids=input_ids.input_ids, max_length=max_length)
print(tokenizer.decode(outputs[0]))