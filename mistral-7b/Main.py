import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import configparser
import TabularToTextualConverter as TabularToTextualConverter
import TextualToTabularConverter as TextualToTabularConverter
import sys

# Append paths if necessary
sys.path.append("./mistral-7b")

device = "cuda" if torch.cuda.is_available() else "cpu"  # Check if CUDA is available

CONFIG_FILE = "../config.ini"

# Set the number of threads for PyTorch
torch.set_num_threads(8)

def main():
    print("Reading configuration file...")
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    dataset = config['dataset']['data_in_use']
    data = config[dataset]['data_dir']

    print("Formatting patient data...")
    patient_data_formatter = TabularToTextualConverter.TabularToTextualConverter(data)
    patient_data_formatter.read_data()
    patient_data_formatter.transform_rows()
    subset_data = patient_data_formatter.get_subset_data(number_of_patients=12)

    print("Loading Mistral-7B Model...")
    tokenizer_path = '/home/mmu/spinning-storage/mmu/mistral-7b/Mistral-7B-Instruct-v0.2'
    model_path = '/home/mmu/spinning-storage/mmu/mistral-7b/Mistral-7B-Instruct-v0.2'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    i = 0
    input_text = config['mistral-7b']['input_text']


    lines_to_skip = ["Here are", "Patient Record", "Prompt:", "Use this:", "Patient i:", "Data:", "Generated Response:"]

    def filter_lines(text: str, prefixes: list) -> str:
        filtered_lines = []
        for line in text.splitlines():
            if not any(line.startswith(prefix) for prefix in prefixes):
                filtered_lines.append(line)
        return "\n".join(filtered_lines)


    while i < len(subset_data):
        print(f"Generating response for patient record {i+1}/{len(subset_data)}...")
        prompt = input_text + str(subset_data[i])

        # Tokenize input text
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate text
        generated_ids = model.generate(inputs['input_ids'], max_length=7000, num_return_sequences=1, no_repeat_ngram_size=2)

        # Decode and convert to text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Store the generated text in a file
        results_txt = config['mistral-7b']['input_file']
        with open(results_txt, 'a') as f:
            content = f"Patient Record {i+1}:\nPrompt: {prompt}\nGenerated Response:\n{generated_text}\n\n"
            filtered_content = filter_lines(content, lines_to_skip)
            f.write(filtered_content + "\n\n")

        i += 1

    print("Converting generated text to tabular format...")
    converter = TextualToTabularConverter.TextualToTabularConverter(CONFIG_FILE)
    converter.process()

if __name__ == "__main__":
    main()
