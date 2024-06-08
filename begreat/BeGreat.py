from be_great import GReaT
import pandas as pd
import configparser

CONFIG_FILE = "../config.ini"

class SyntheticDataGeneratorBeGreat:
    def __init__(self):
        print("Reading configuration file...")
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE)

        self.llm = str(config['begreat']['llm'])
        self.batch_size = int(config['begreat']['batch_size'])
        self.epochs = int(config['begreat']['epochs'])
        self.save_steps = int(config['begreat']['save_steps'])
        self.data = pd.read_csv(config['dataset']['dataset_dir'])
        self.n_samples = int(config['begreat']['n_samples']) 
        self.output_file = config['begreat']['output_file']

        self.model = GReaT(llm=self.llm, batch_size=self.batch_size, epochs=self.epochs, save_steps=self.save_steps)


    def fit(self):
        print("Fitting the model to the data...")
        self.model.fit(self.data)
    
    def generate_samples(self):
        print("Generating synthetic data...")
        synthetic_data = self.model.sample(n_samples=self.n_samples)
        return synthetic_data
    
    def save_samples(self, synthetic_data):
        synthetic_data.to_csv(self.output_file, index=False)
        print(f"Synthetic data generated and saved to {self.output_file}")