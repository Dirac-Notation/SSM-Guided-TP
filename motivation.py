import torch
import json
import math
import os
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import AutoTokenizer

from eagle.model.ea_model import EaModel
from eagle.model.kv_cache import initialize_past_key_values

base_model_path = "meta-llama/Llama-2-7b-chat-hf"
EAGLE_model_path = "yuhuili/EAGLE-llama2-chat-7B"
dataset = "fewshot_data/cnn_dailymail-3shot.jsonl"

with open(dataset, "r") as f:
    prompts = []
    answers = []
    
    datalines = f.readlines()
    articles = []
    
    for dataline in datalines:
        articles.append(json.loads(dataline))
        
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    for data in tqdm(articles, desc="Tokenizing"):
        input = data["article"]
        answer = data["summary_gt"]
        
        input_ids = tokenizer(input, return_tensors="pt").input_ids

        prompts.append(input_ids)
        answers.append(answer)
    
    del tokenizer

model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    torch_dtype=torch.float16,
    device_map=f"cuda:5",
).eval()

input_ids = prompts[0].to(model.base_model.device)

past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model.base_model)

with torch.inference_mode():
    outputs_ltm = model.base_model.model(input_ids, past_key_values=past_key_values, output_attentions=True)

    last_hidden_states = outputs_ltm.last_hidden_state[:,:-1,:]
    attentions_ltm = torch.stack([tensor.cpu() for tensor in outputs_ltm.attentions]).squeeze(1)

    outputs_ssm = model.ea_layer(last_hidden_states, input_ids[:,1:], output_attentions=True)
    
    logits_ssm = model.base_model.lm_head(outputs_ssm[0])
    attentions_ssm = torch.stack([tensor.cpu() for tensor in outputs_ssm[1]]).squeeze(1)

attention_ltm = attentions_ltm.mean(dim=(0,1))[1:,1:]
attention_ssm = attentions_ssm.mean(dim=(0,1))

import pdb; pdb.set_trace()

plt.imshow(attention_ltm.pow(0.2), cmap="Blues", interpolation="nearest")
plt.savefig("ltm_average.png")
plt.close()

plt.imshow(attention_ssm.pow(0.2), cmap="Blues", interpolation="nearest")
plt.savefig("ssm_average.png")
plt.close()