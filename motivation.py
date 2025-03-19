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

from utils import sim, load_datasets, diff

base_model_path = "meta-llama/Llama-2-7b-chat-hf"
EAGLE_model_path = "yuhuili/EAGLE-llama2-chat-7B"
dataset = "fewshot_data/cnn_dailymail-0shot.jsonl" # fewshot_data/multi_news-0shot.jsonl fewshot_data/cnn_dailymail-0shot.jsonl

model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    torch_dtype=torch.float16,
    device_map=f"cuda",
).eval()

prompts, answers = load_datasets(dataset, model.tokenizer)

results = {}
for layer in range(32):
    for head in range(32):
        results[f"{layer}_{head}"] = [0, 0, 0, 0, 0, 0, 0, 0]

for prompt in tqdm(prompts):
    input_ids = prompt.to(model.base_model.device)

    past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model.base_model)

    with torch.inference_mode():
        outputs_ltm = model.base_model.model(input_ids, past_key_values=past_key_values, output_attentions=True)

        last_hidden_states = outputs_ltm.last_hidden_state[:,:-1,:]
        attentions_ltm = torch.stack([tensor.cpu() for tensor in outputs_ltm.attentions]).squeeze(1)

        outputs_ssm = model.ea_layer(last_hidden_states, input_ids[:,1:], output_attentions=True)
        
        logits_ssm = model.base_model.lm_head(outputs_ssm[0])
        attentions_ssm = torch.stack([tensor.cpu() for tensor in outputs_ssm[1]]).squeeze(1)

    iteration = 10
    for layer in range(32):
        for head in range(32):
            for i in range(1,iteration+1):
                union_12, union_13, union_23, union_all, score_union_12, score_union_13, score_union_all, all_score = diff(
                    attentions_ltm[layer,head,-i,1:],
                    attentions_ltm[layer,head,:-i,1:].sum(dim=-2),
                    attentions_ssm[:,:,-i].sum(dim=(0,1)),
                    100
                )
                results[f"{layer}_{head}"][0] += union_12/iteration
                results[f"{layer}_{head}"][1] += union_13/iteration
                results[f"{layer}_{head}"][2] += union_23/iteration
                results[f"{layer}_{head}"][3] += union_all/iteration
                
                results[f"{layer}_{head}"][4] += score_union_12/iteration
                results[f"{layer}_{head}"][5] += score_union_13/iteration
                results[f"{layer}_{head}"][6] += score_union_all/iteration
                
                results[f"{layer}_{head}"][7] += all_score/iteration

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
