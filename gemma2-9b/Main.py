
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import TabularToTextualConverter as TabularToTextualConverter
import TextualToTabularConverter as TextualToTabularConverter
import sys
import configparser
from typing import List, Optional

sys.path.append("./gemma2-9b")
CONFIG_FILE = "../config.ini"

# Set CUDA_LAUNCH_BLOCKING for debugging
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Enable device-side assertions

if __name__ == "__main__":
    try:
        print("Reading configuration file...")
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE)
        dataset = config['dataset']['data_in_use']
        data = config[dataset]['data_dir']

        print("Formatting patient data...")
        patient_data_formatter = TabularToTextualConverter.PatientDataFormatter(data)
        patient_data_formatter.read_data()
        patient_data_formatter.transform_rows()
        #combined_string = patient_data_formatter.get_combined_string()

        # create a subset of the data (every 15 patients)
        subset_data = patient_data_formatter.get_subset_data(number_of_patients=12)
        
        # Load the model
        print("Loading model...")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        tokenizer = AutoTokenizer.from_pretrained("/home/mmu/spinning-storage/mmu/gemma2/gemma-2-9b-it/")
        model = AutoModelForCausalLM.from_pretrained("/home/mmu/spinning-storage/mmu/gemma2/gemma-2-9b-it", quantization_config=quantization_config)

        # Adjust max_length for longer sequences
        max_length = 5000  # Increase this value as needed

        #output_file = "results/synthetic_data_gemma2_9b.txt"
        results_txt = config['gemma2-9b']['input_file']

        i = 0

        def filter_lines(text: str, prefixes: List[str]) -> str:
            filtered_lines = []
            for line in text.splitlines():
                if not any(line.startswith(prefix) for prefix in prefixes):
                    filtered_lines.append(line)
            return "\n".join(filtered_lines)
        
        # Store the results in a txt file
        lines_to_skip = ["Generate", "Disease:", "Use this", "\" Generate 12"]
        with open(results_txt, "a") as f:  # Open in append mode to avoid overwriting

            while i < len(subset_data):
                input_text = config['gemma2-9b']['input_text_1']

                print("Subset number: " + str(i + 1) + " out of " + str(len(subset_data)))
                # Use the model
                input_text = input_text + "\n" + str(subset_data[i]) + "\n"
                
                input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

                print("Generating patient records...")
                outputs = model.generate(input_ids=input_ids.input_ids, max_length=max_length)

                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                filtered_content = filter_lines(generated_text, lines_to_skip)
                
                f.write(filtered_content + "\n\n")

                i += 1

        print("Storing the results in txt file...")
        textualToTabularConverter = TextualToTabularConverter.TextualToTabularConverter("results/synthetic_data_gemma2_9b.txt")
        textualToTabularConverter.write_to_csv("results/synthetic_data_gemma2_9b.csv")
    except Exception as e:
        print(e)
        raise e
