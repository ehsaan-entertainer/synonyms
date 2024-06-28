from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import transformers
import torch
import os

from huggingface_hub import login

# Replace 'your_access_token' with your actual access token
cache_dir = "/Users/ehsaani/codebase/synonyms/models"
access_token = "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"
login(token=access_token)

# huggingface-cli download meta-llama/Meta-Llama-3-8B --cache-dir /Users/ehsaani/codebase/synonyms/models

def load_llama3_8b(model_id="meta-llama/Meta-Llama-3-8B", word="pizza"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)

    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )

    # Test the pipeline
    output = text_generator(f"Synonyms of {word}", max_length=50)
    print(output)


load_llama3_8b()
print('sfdlksdflksd')

