from ctgan import CTGAN
import pandas as pd
import configparser

CONFIG_FILE = "../config.ini"

class SyntheticDataGeneratorCTGAN:
    def __init__(self, epochs=10):
        self.ctgan = CTGAN(epochs=epochs)
        print("Reading configuration file...")
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE)
        self.output_file = config['ctgan']['output_file']
        self.data = pd.read_csv(config['dataset']['dataset_dir'])
        self.discrete_columns = config['ctgan']['discrete_columns'].split(', ')
        self.num_samples = int(config['ctgan']['num_samples'])
    
    def fit_and_generate(self):

        print("Fitting the data using CTGAN...")
        self.ctgan.fit(self.data, self.discrete_columns)
        synthetic_data = self.ctgan.sample(self.num_samples)
        synthetic_data.to_csv(self.output_file, index=False)
        print("Synthetic data generated and saved to", self.output_file)

# Measure the convergence


generator = SyntheticDataGeneratorCTGAN(epochs=10)
generator.fit_and_generate()
