from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import TabularToTextualConverter as TabularToTextualConverter
import TextualToTabularConverter as TextualToTabularConverter
import sys

sys.path.append("./gemma-7b")
DATA = "../datasets/data.csv"


if __name__ == "__main__":
    patient_data_formatter = TabularToTextualConverter.PatientDataFormatter(DATA)
    patient_data_formatter.read_data()
    patient_data_formatter.transform_rows()
    combined_string = patient_data_formatter.get_combined_string()

    # create a subset of the data (every 15 patients)
    subset_data = patient_data_formatter.get_subset_data()
    
    # Load the model
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained("/home/mmu/spinning-storage/mmu/gemma-7b/")
    model = AutoModelForCausalLM.from_pretrained("/home/mmu/spinning-storage/mmu/gemma-7b/", quantization_config=quantization_config)



    # Adjust max_length for longer sequences
    max_length = 5000  # Increase this value as needed


    output_file = "results/synthetic_data_gemma_7b.txt"

    i = 0


    
    with open(output_file, "a") as f:  # Open in append mode to avoid overwriting

        while i < len(subset_data):
            print("Subset number" + str(i))
            # Use the model
            input_text = " Generate 12 additional patient records in the following format:\
                            Disease: disease, Fever: fever, Cough: cough, Fatigue: fatigue, Difficulty Breathing: difficulty_breathing, Age: age, Gender: gender, Blood Pressure: blood_pressure, Cholesterol Level: cholesterol_level, Outcome Variable: outcome\
                            Use this current data for reference:\
                            " + str(subset_data[i]) + "\n"
                            
            input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

            outputs = model.generate(input_ids=input_ids.input_ids, max_length=max_length)

                
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)


            f.write(generated_text)
            f.write("\n")

            i += 1

            
    textualToTabularConverter = TextualToTabularConverter.TextualToTabularConverter("results/synthetic_data_gemma_7b.txt")
    textualToTabularConverter.write_to_csv("results/synthetic_data_gemma_7b.csv")