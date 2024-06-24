import csv
import re

class TextualToTabularConverter:
    def __init__(self, data):
        self.data = data
        self.header = ["Patient i", "Disease", "Fever", "Cough", "Fatigue", "Difficulty Breathing", "Age", "Gender", "Blood Pressure", "Cholesterol Level", "Outcome Variable"]
        self.pattern_multiline = re.compile(
            r"\*\*Patient\s+(\d+):\*\*\s*"
            r"Disease:\s*(.+?)\s*"
            r"Fever:\s*(Yes|No)\s*"
            r"Cough:\s*(Yes|No)\s*"
            r"Fatigue:\s*(Yes|No)\s*"
            r"Difficulty Breathing:\s*(Yes|No)\s*"
            r"Age:\s*(\d+)\s*"
            r"Gender:\s*(Male|Female)\s*"
            r"Blood Pressure:\s*(Low|Normal|High)\s*"
            r"Cholesterol Level:\s*(Low|Normal|High)\s*"
            r"Outcome Variable:\s*(Positive|Negative)"
        )
        self.pattern_singleline = re.compile(
            r"\*\*Patient\s+(\d+):\*\*\s*"
            r"Disease:\s*(.+?),\s*"
            r"Fever:\s*(Yes|No),\s*"
            r"Cough:\s*(Yes|No),\s*"
            r"Fatigue:\s*(Yes|No),\s*"
            r"Difficulty Breathing:\s*(Yes|No),\s*"
            r"Age:\s*(\d+),\s*"
            r"Gender:\s*(Male|Female),\s*"
            r"Blood Pressure:\s*(Low|Normal|High),\s*"
            r"Cholesterol Level:\s*(Low|Normal|High),\s*"
            r"Outcome Variable:\s*(Positive|Negative)"
        )

    def extract_data(self):
        matches_multiline = re.findall(self.pattern_multiline, self.data)
        print(matches_multiline)
        matches_singleline = re.findall(self.pattern_singleline, self.data)
        print(matches_singleline)

        return matches_multiline + matches_singleline

    def write_to_csv(self, filename):
        data = self.extract_data()
        with open(filename, "w", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.header)
            csvwriter.writerows(data)
        print(f"CSV file '{filename}' has been created.")
