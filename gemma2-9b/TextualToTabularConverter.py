import csv
import re

class TextualToTabularConverter:
    def __init__(self, filename):
        self.filename = filename
        self.header = ["Patient ID", "Disease", "Fever", "Cough", "Fatigue", "Difficulty Breathing", "Age", "Gender", "Blood Pressure", "Cholesterol Level", "Outcome Variable"]
        self.pattern = re.compile(
            r"Patient (\d+): \[Disease: ([^,]+), Fever: (Yes|No), Cough: (Yes|No), Fatigue: (Yes|No), Difficulty Breathing: (Yes|No), Age: (\d+), Gender: (Male|Female), Blood Pressure: (Low|Normal|High), Cholesterol Level: (Low|Normal|High), Outcome Variable: (Positive|Negative)\]"
        )

    def read_data(self):
        with open(self.filename, 'r') as file:
            data = file.read()
        return data

    def extract_data(self):
        data = self.read_data()
        matches = re.findall(self.pattern, data)
        return matches

    def write_to_csv(self, output_filename):
        data = self.extract_data()
        with open(output_filename, "w", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.header)
            csvwriter.writerows(data)
        print(f"CSV file '{output_filename}' has been created.")