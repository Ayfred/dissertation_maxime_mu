# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional
import fire
import sys
import TabularToTextualConverter as TabularToTextualConverter
import configparser

sys.path.append("./llama")
from llama import Llama, Dialog

CONFIG_FILE = "../config.ini"

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """

    print("Reading configuration file...")
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    data = config['dataset']['dataset_dir']

    print("Formatting patient data...")
    patient_data_formatter = TabularToTextualConverter.TabularToTextualConverter(data)
    patient_data_formatter.read_data()
    patient_data_formatter.transform_rows()
    #combined_string = patient_data_formatter.get_combined_string()

    subset_data = patient_data_formatter.get_subset_data(number_of_patients=5)

    print("Building Llama generator...")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    i = 0

    input_text = config['llama-2-7b']['input_text']

    while i < len(subset_data):
        print("Generating patient records...")
        dialogs: List[Dialog] = [
            [{"role": "user", "content": input_text +  str(subset_data[i])}],
        ]



        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        """
        print("Printing generated records...")
        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            print("\n==================================\n")
        """

        results_txt = config['llama-2-7b']['input_file']

        # Store the results in a txt file
        print("Storing the results in txt file...")
        with open(results_txt, 'a') as f:
            for dialog, result in zip(dialogs, results):
                for msg in dialog:
                    f.write(f"{msg['role'].capitalize()}: {msg['content']}\n")
                f.write(
                    f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                )
                f.write("\n")

        i += 1
        if i == 10:
            break

if __name__ == "__main__":
    fire.Fire(main)
