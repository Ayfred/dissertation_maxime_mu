import csv
import re

class TextualToTabularConverter:
    def __init__(self, data):
        self.data = data
        self.header = ["Patient i", "Disease", "Fever", "Cough", "Fatigue", "Difficulty Breathing", "Age", "Gender", "Blood Pressure", "Cholesterol Level", "Outcome Variable"]
        self.pattern = re.compile(
                    r"\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(Yes|No)\s*\|\s*(Yes|No)\s*\|\s*(Yes|No)\s*\|\s*(Yes|No)\s*\|\s*(\d+)\s*\|\s*(Male|Female)\s*\|\s*(Low|Normal|High)\s*\|\s*(Low|Normal|High)\s*\|\s*(Positive|Negative)\s*\|",
                    re.DOTALL
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