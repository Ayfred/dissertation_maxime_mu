from ctgan import CTGAN
import pandas as pd


data = pd.read_csv('data.csv')    

# Names of the columns that are discrete
discrete_columns = [
    'Disease',
    'Fever',
    'Cough',
    'Fatigue',
    'Difficulty Breathing',
    'Gender',
    'Blood Pressure',
    'Cholesterol Level',
    'Outcome Variable'
]

ctgan = CTGAN(epochs=10)
ctgan.fit(data, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(1000)

print(synthetic_data)

# Save the synthetic data to a CSV file
synthetic_data.to_csv('ctgan/results/synthetic_data.csv', index=False)