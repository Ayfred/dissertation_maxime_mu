import csv

# Open the input and output files
with open('gemma/results/output.txt', 'r') as input_file, open('gemma/results/patients.csv', 'w', newline='') as output_file:
    # Create a CSV writer object
    writer = csv.writer(output_file)
    
    # Write the header row
    writer.writerow(['Patient ID', 'Disease', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 
                     'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Outcome Variable'])
    
    # Skip the first 3 lines as they are headers in the text file
    for _ in range(3):
        next(input_file)
    
    # Process each line in the input file
    for line in input_file:
        # Split the line by '|' character and strip whitespace
        fields = [field.strip() for field in line.split('|')]
        # Remove empty strings from the fields list
        fields = [field for field in fields if field]
        # Write the fields to the CSV file
        writer.writerow(fields)

print("Conversion completed successfully.")
