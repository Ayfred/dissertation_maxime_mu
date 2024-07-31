import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
        "text-generation",
        model="/home/mmu/spinning-storage/mmu/llama3.1/meta-llama/Meta-Llama-3.1-8B-Instructhuggingface-cli/original",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda"
        )




