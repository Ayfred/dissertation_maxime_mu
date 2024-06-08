import csv
import re

class TextualToTabularConverter:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.data = ""
        self.matches = []
        self.headers = ['Disease', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Outcome Variable']
        self.pattern = re.compile(
            r'Patient \d+: \[Disease: (.*?), Fever: (.*?), Cough: (.*?), Fatigue: (.*?), Difficulty Breathing: (.*?), Age: (\d+), Gender: (.*?), Blood Pressure: (.*?), Cholesterol Level: (.*?), Outcome Variable: (.*?)\]'
        )

    def read_data(self):
        with open(self.input_file, 'r') as file:
            self.data = file.read()

    def parse_data(self):
        self.matches = self.pattern.findall(self.data)

    def write_csv(self):
        with open(self.output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.headers)
            for match in self.matches:
                writer.writerow(match)

    def process(self):
        self.read_data()
        self.parse_data()
        self.write_csv()