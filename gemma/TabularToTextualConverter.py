import pandas as pd

class PatientDataFormatter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.string_list = []

    def read_data(self):
        self.data = pd.read_csv(self.file_path)
    
    def transform_rows(self):
        # Check if data is read
        if self.data is None:
            raise ValueError("Data not loaded. Call read_data() first.")
        
        # Iterate over each row in the DataFrame
        for index, row in self.data.iterrows():
            # Create a list to store the key-value pairs
            key_value_pairs = []
            
            # Iterate over each column in the row
            for col_name in self.data.columns:
                key_value_pairs.append(f"{col_name}: {row[col_name]}")
            
            # Join the key-value pairs into a single string
            row_string = ", ".join(key_value_pairs)
            
            # Prepend the patient number and add to the list
            patient_string = f"Patient {index + 1}: [{row_string}]"
            self.string_list.append(patient_string)

    def get_string_list(self):
        return self.string_list

    def get_combined_string(self):
        # Combine the list into a single string
        return ", ".join(self.string_list)

    def print_string_list(self):
        for s in self.string_list:
            print(s)

    def print_combined_string(self):
        print(self.get_combined_string())

    def get_subset_data(self, number_of_patients=15):
        # Subdivide the string list into subsets
        subset_data = []
        
        # Iterate over the string list
        for i in range(0, len(self.string_list), number_of_patients):
            subset_data.append(self.string_list[i:i+number_of_patients])

        return subset_data




if __name__ == "__main__":
    file_path = "datasets/data.csv"
    
    formatter = PatientDataFormatter(file_path)
    formatter.read_data()
    formatter.transform_rows()

    # Print the combined string
    combined_string = formatter.get_combined_string()
    
