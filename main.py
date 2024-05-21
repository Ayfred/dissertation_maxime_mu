import gemma.gemma as gemma

MODEL_2B_IT = "2b-it"
MODEL_7B_IT = "7b-it"

class Main:
    def __init__(self, model):
        self.command = "./gemma/gemma.cpp/build/gemma"
        self.model = model
        self.gemma_model = None

    def setup(self):
        self.gemma_model = gemma.GemmaModel(self.command, self.model)
        self.gemma_model.start_process()

    def run(self, input_text):
        output = self.gemma_model.run_model(input_text)
        self.gemma_model.write_output_to_file(output)
        self.gemma_model.print_output(output)


if __name__ == "__main__":
    main_app = Main(MODEL_2B_IT)
    main_app.setup()

    input_text = " \
                  patient 1 : [disease: Asthama, age 25, Male], patient 2 : [disease: Asthma, age 30, Female], patient 3 : [disease: Eczema, age 55, Female], patient 4 : [disease: ADHD, age 64, Male] \
                  Question: What disease will patient 4 have?\n"
    main_app.run(input_text)
