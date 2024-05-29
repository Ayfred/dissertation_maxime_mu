import subprocess

MODEL_2B_IT = "2b-it"
MODEL_7B_IT = "7b-it"

class GemmaModel:

    def __init__(self, command, model):
        self.command = command
        self.model = model
        self.output = ""
        if self.model == MODEL_2B_IT:
            self.args = [
                "--tokenizer", "./gemma/gemma.cpp/build/tokenizer.spm",
                "--compressed_weights", "./gemma/gemma.cpp/build/2b-it-sfp.sbs",
                "--max_tokens", "4000",
                "--model", "2b-it"
            ]
        elif self.model == MODEL_7B_IT:
            self.args = [
                "--tokenizer", "./gemma/gemma.cpp/build/7b/tokenizer.spm",
                "--compressed_weights", "./gemma/gemma.cpp/build/7b/7b-it-sfp.sbs",
                "--model", "7b-it"
            ]
        else:
            raise ValueError("Invalid model. Must be one of '2b-it' or '7b-it'.")

        self.process = None

    def start_process(self):
        self.process = subprocess.Popen(
            [self.command] + self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    def run_model(self, input_text):
        if not self.process:
            raise RuntimeError("Process not started. Call start_process() first.")
        
        self.process.stdin.write(input_text)
        self.process.stdin.flush()
        
        output, error = self.process.communicate()
        
        if self.model == MODEL_2B_IT:
            # Remove unwanted parts of the output
            output = output.split("How are you doing today? Is there anything I can help you with?")[0]
            output = output.split(">")[1].strip()
        
        self.output = output
        return output

    def write_output_to_file(self, output, filename='gemma/results/output.txt'):
        with open(filename, 'w') as f:
            f.write(output)

    def print_output(self, output):
        print(output)

    def get_output(self):
        return self.output

