# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional
import fire
import sys
import TabularToTextualConverter as TabularToTextualConverter

sys.path.append("./llama3")
from llama import Dialog, Llama

DATA = "../datasets/data.csv"


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """

    patient_data_formatter = TabularToTextualConverter.PatientDataFormatter(DATA)
    patient_data_formatter.read_data()
    patient_data_formatter.transform_rows()
    combined_string = patient_data_formatter.get_combined_string()

    print(len(combined_string))
    subset_data = patient_data_formatter.get_subset_data(number_of_patients=5)
    print(len(subset_data))


    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = [
        [{"role": "user", "content": "Generate 50 additional patient records in the following format and don't hesitate to generate new diseases as well:\
                Patient i: [Disease: disease, Fever: fever, Cough: cough, Fatigue: fatigue, Difficulty Breathing: difficulty_breathing, Age: age, Gender: gender, Blood Pressure: blood_pressure, Cholesterol Level: cholesterol_level, Outcome Variable: outcome]\
                Use this current data for reference:\
                Data: " +  str(subset_data[0])}],
        
    ]
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")
    
    # Store the results in a txt file
    with open('results/synthetic_data.txt', 'w') as f:
        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                f.write(f"{msg['role'].capitalize()}: {msg['content']}\n")
            f.write(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            f.write("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)
