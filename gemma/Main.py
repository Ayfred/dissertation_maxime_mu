import Gemma as Gemma
import pandas as pd
import TabularToTextualConverter as TabularToTextualConverter
import TextualToTabularConverter as TextualToTabularConverter


MODEL_2B_IT = "2b-it"
MODEL_7B_IT = "7b-it"
DATA = "../datasets/data.csv"

class Main:
    def __init__(self, model):
        self.command = "./gemma.cpp/build/gemma"
        self.model = model
        self.gemma_model = None

    def setup(self):
        self.gemma_model = Gemma.GemmaModel(self.command, self.model)
        self.gemma_model.start_process()

    def run(self, input_text):
        output = self.gemma_model.run_model(input_text)
        self.gemma_model.write_output_to_file(output)
        self.gemma_model.print_output(output)

    def get_output(self):
        return self.gemma_model.get_output()




if __name__ == "__main__":
    patient_data_formatter = TabularToTextualConverter.PatientDataFormatter(DATA)
    patient_data_formatter.read_data()
    patient_data_formatter.transform_rows()
    combined_string = patient_data_formatter.get_combined_string()

    # create a subset of the data (every 15 patients)
    subset_data = patient_data_formatter.get_subset_data()
    
    #main_app = Main(MODEL_2B_IT)
    #main_app.setup()

    output_file = "gemma/results/output.txt"

    i = 0


    """
    with open(output_file, "a") as f:  # Open in append mode to avoid overwriting

        while i < len(subset_data):
            main_app = Main(MODEL_2B_IT)
            main_app.setup()


            #input_text = "Question: Can you generate more patients based on the following data? \
            #    Put it in the following format: Patient i: [Disease: disease, Fever: fever, Cough: cough, Fatigue: fatigue, Difficulty Breathing: difficulty_breathing, Age: age, Gender: gender, Blood Pressure: blood_pressure, Cholesterol Level: cholesterol_level, Outcome Variable: outcome] \
            #    Data: " + str(subset_data[i]) + "\n"
            
            input_text = " Generate additional patient records in the following format:\
                Patient i: [Disease: disease, Fever: fever, Cough: cough, Fatigue: fatigue, Difficulty Breathing: difficulty_breathing, Age: age, Gender: gender, Blood Pressure: blood_pressure, Cholesterol Level: cholesterol_level, Outcome Variable: outcome]\
                Use this current data for reference:\
                Data: " + str(subset_data[i]) + "\n"
                

            main_app.run(input_text)

            f.write(main_app.get_output())
            f.write("\n")

            i += 1

            if i == 1:
                break
    
            
    textualToTabularConverter = TextualToTabularConverter.TextualToTabularConverter(output_file)
    textualToTabularConverter.write_to_csv("gemma/results/synthetic_data_2b_it.csv")

    """

    main_app = Main(MODEL_2B_IT)
    main_app.setup()
    
    input_text = " Hi what is your name ?"
    main_app.run(input_text)

    print(main_app.get_output())


