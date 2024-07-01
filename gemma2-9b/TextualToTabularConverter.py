import csv
import re

class TextualToTabularConverter:
    def __init__(self, data):
        self.data = data
        self.header = ["Patient ID", "Disease", "Fever", "Cough", "Fatigue", "Difficulty Breathing", "Age", "Gender", "Blood Pressure", "Cholesterol Level", "Outcome Variable"]
        self.pattern = re.compile(
            r"Patient (\d+): \[Disease: (.+?), Fever: (Yes|No), Cough: (Yes|No), Fatigue: (Yes|No), Difficulty Breathing: (Yes|No), Age: (\d+), Gender: (Male|Female), Blood Pressure: (Low|Normal|High), Cholesterol Level: (Low|Normal|High), Outcome Variable: (Positive|Negative)\]"
        )

    def extract_data(self):
        matches = re.findall(self.pattern, self.data)
        return matches

    def write_to_csv(self, filename):
        data = self.extract_data()
        with open(filename, "w", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.header)
            csvwriter.writerows(data)
        print(f"CSV file '{filename}' has been created.")