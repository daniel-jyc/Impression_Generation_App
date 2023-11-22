import streamlit as st
import sns
import os
import torch
import logging
import numpy as np
import pandas as pd

from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM

sns.set()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

base_model, new_model = "microsoft/biogpt" , 'rlmjy/ft_new_biogpt'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

finetuned_model = AutoPeftModelForCausalLM.from_pretrained(
    new_model,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    # from_tf=True, #just adding this because the error said so
    # device_map="auto" #removing this because it makes error
    device_map={"":0}
)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

generation_config = GenerationConfig(
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.25,
    do_sample = True
)

def generate_prompt_test(input):
    
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{"Find the impresssion from the background and findings obtained in the radiology examination"}
### Input:
{input}
### Response: """

def get_output(s):
    text = s.split("Response:")[-1].split("</s>")[0].strip()
    return(text)

def make_inference(model_a, instruction, context = None):
    
    prompt = generate_prompt_test(input)
        
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to("cuda:0")
    model_a.generation_config.pad_token_id = model_a.generation_config.eos_token_id

    outputs_1 = model_a.generate(**inputs, max_new_tokens=128, #generation_config = generation_config,
                                early_stopping=True,
                                num_beams=3,
                                num_return_sequences=3,)
   
    return get_output(tokenizer.decode(outputs_1[0], skip_special_tokens=True))

def main():
    # Set the title of the Streamlit app
    st.title("Radiology Impression Generation")

    background = st.text_input('Clinical Background', "Fill in clinical background", key = "placeholder",)
    findings = st.text_input('Radiology Findings', "Fill in radiology findings", key = "placeholder",)
    input = background + findings

    if st.button('Predict'):
        text = make_inference(finetuned_model, input)

        st.write("## Generated Impression:", text)

if __name__ == '__main__':
    main()