import torch
import json
import math
import os
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import sim, load_datasets, diff

ltm_model_path = "facebook/opt-6.7b"
ssm_model_path = "facebook/opt-125m"
dataset = "fewshot_data/cnn_dailymail-0shot.jsonl" # fewshot_data/multi_news-0shot.jsonl fewshot_data/cnn_dailymail-0shot.jsonl

device = torch.device("cuda:3")
tokenizer = AutoTokenizer.from_pretrained(ltm_model_path)
model_ltm = AutoModelForCausalLM.from_pretrained(ltm_model_path, torch_dtype=torch.float16).to(device=device).eval()
model_ssm = AutoModelForCausalLM.from_pretrained(ssm_model_path, torch_dtype=torch.float16).to(device=device).eval()

prompts, answers = load_datasets(dataset, tokenizer)

results = {}
for layer in range(32):
    for head in range(32):
        results[f"{layer}_{head}"] = [0, 0, 0, 0, 0, 0, 0, 0]

for prompt in tqdm(prompts):
    input_ids = prompt.to(device)

    with torch.inference_mode():
        outputs_ltm = model_ltm(input_ids, output_attentions=True)

        attentions_ltm = torch.stack([tensor.cpu() for tensor in outputs_ltm.attentions]).squeeze(1)

        outputs_ssm = model_ssm(input_ids, output_attentions=True)
        
        attentions_ssm = torch.stack([tensor.cpu() for tensor in outputs_ssm.attentions]).squeeze(1)

    iteration = 10
    for layer in range(32):
        for head in range(32):
            for i in range(1,iteration+1):
                # union_12, union_13, union_23, union_all, score_union_12, score_union_13, score_union_all, all_score
                tmp_result = diff(
                    attentions_ltm[layer,head,-i,1:],
                    attentions_ltm[layer,head,:-i,1:].sum(dim=-2),
                    attentions_ssm[:,:,-i].sum(dim=(0,1)),
                    100
                )
                
                for idx, tmp in enumerate(tmp_result):
                    results[f"{layer}_{head}"][idx] += tmp/iteration

value_12 = 0
value_13 = 0
value_23 = 0
value_all = 0

score_12 = 0
score_13 = 0
score_all = 0

all_score = 0

for layer in range(32):
    for head in range(32):
        value_12 += results[f"{layer}_{head}"][0]
        value_13 += results[f"{layer}_{head}"][1]
        value_23 += results[f"{layer}_{head}"][2]
        value_all += results[f"{layer}_{head}"][3]
        
        score_12 += results[f"{layer}_{head}"][4]
        score_13 += results[f"{layer}_{head}"][5]
        score_all += results[f"{layer}_{head}"][6]
        
        all_score += results[f"{layer}_{head}"][7]

value_12 /= len(prompts)*1024
value_13 /= len(prompts)*1024
value_23 /= len(prompts)*1024
value_all /= len(prompts)*1024

score_12 /= len(prompts)*1024
score_13 /= len(prompts)*1024
score_all /= len(prompts)*1024

all_score /= len(prompts)*1024

print(value_12, value_13, value_23, value_all)
print(score_12, score_13, score_all)
print(all_score)

# attention_ltm = attentions_ltm.mean(dim=(0,1))[1:,1:]
# attention_ssm = attentions_ssm.mean(dim=(0,1))

# plt.title("LTM")
# plt.matshow(attention_ltm.pow(0.2), cmap="Blues")
# plt.savefig("LTM.png")
# plt.close()

# plt.title("SSM")
# plt.matshow(attention_ssm.pow(0.2), cmap="Blues")
# plt.savefig("SSM.png")
# plt.close()
