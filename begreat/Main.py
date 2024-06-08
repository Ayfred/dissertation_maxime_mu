import BeGreat as bg

if __name__ == "__main__":
    generator = bg.SyntheticDataGeneratorBeGreat()
    
    generator.fit()
    synthetic_data = generator.generate_samples()
    generator.save_samples(synthetic_data)