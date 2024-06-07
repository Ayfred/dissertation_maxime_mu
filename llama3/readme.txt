command line for generating new data using Llama3 8B instruct

torchrun --nproc_per_node 1 Main.py     --ckpt_dir llama3/Meta-Llama-3-8B-Instruct/     --tokenizer_path llama3/Meta-Llama-3-8B-Instruct/tokenizer.model     --max_seq_len 8192 --max_batch_size 6
