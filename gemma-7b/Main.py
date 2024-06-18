from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Load the model
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("/home/mmu/spinning-storage/mmu/gemma-7b/")
model = AutoModelForCausalLM.from_pretrained("/home/mmu/spinning-storage/mmu/gemma-7b/", quantization_config=quantization_config)

# Use the model
input_text = "What is ChatGPT ?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")


# Adjust max_length for longer sequences
max_length = 1000  # Increase this value as needed
outputs = model.generate(input_ids=input_ids.input_ids, max_length=max_length)

# Decode and print the generated output
print("---------------------------------------------------------")
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
print("---------------------------------------------------------")
