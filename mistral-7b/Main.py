import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import configparser
import TabularToTextualConverter as TabularToTextualConverter
import TextualToTabularConverter as TextualToTabularConverter
import sys

sys.path.append("./mistral-7b")

device = "cuda" # the device to load the model onto
CONFIG_FILE = "../config.ini"

# Set the number of processes
torch.set_num_threads(8)


print("Reading configuration file...")
config = configparser.ConfigParser()
config.read(CONFIG_FILE)
dataset = config['dataset']['data_in_use']
data = config[dataset]['data_dir']

print("Formatting patient data...")
patient_data_formatter = TabularToTextualConverter.TabularToTextualConverter(data)
patient_data_formatter.read_data()
patient_data_formatter.transform_rows()
#combined_string = patient_data_formatter.get_combined_string()

subset_data = patient_data_formatter.get_subset_data(number_of_patients=5)

print("Loading Mistral-7b Model...")
tokenizer = AutoTokenizer.from_pretrained('/home/support/llm/Mistral-7B-Instruct-v0.2')
model = AutoModelForCausalLM.from_pretrained('/home/support/llm/Mistral-7B-Instruct-v0.2')

i = 0

input_text = config['mistral-7b']['input_text']

while i < len(subset_data):
    print("Generating patient records...")


    messages = [
        {"role": "user", "content":input_text + str(subset_data[i])}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)


    results_txt = config['mistral-7b']['input_file']

    # Store the results in a txt file
    print("Storing the results in txt file...")
    with open(results_txt, 'a') as f:
        for result in decoded:
            f.write(result + '\n')

        f.write("\n")



    i += 1
    if i == 1:
        break

print("Converting generated text to tabular format...")
converter = TextualToTabularConverter.TextualToTabularConverter(CONFIG_FILE)
converter.process()
print(decoded[0])