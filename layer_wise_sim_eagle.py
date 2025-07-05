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

from utils import sim, load_datasets, diff, cosine_sim

def main(args):
    model_name = "llama"
    
    if model_name == "llama":
        base_model_path = "meta-llama/Llama-2-7b-chat-hf"
        EAGLE_model_path = "yuhuili/EAGLE-llama2-chat-7B"
    elif model_name == "vicuna":
        base_model_path = "lmsys/vicuna-7b-v1.5"
        EAGLE_model_path = "yuhuili/EAGLE-Vicuna-7B-v1.3"
        
    dataset = "fewshot_data/cnn_dailymail-0shot.jsonl"

    device_map = f"cuda:{args.gpu}"
    
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=EAGLE_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    ).eval()

    prompts, answers = load_datasets(dataset, model.tokenizer)

    results_h2o = [0 for _ in range(32)]
    results_ssm = [0 for _ in range(32)]
    
    for prompt in tqdm(prompts):
        input_ids = prompt.to(model.base_model.device)

        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model.base_model)

        with torch.inference_mode():
            outputs_ltm = model.base_model.model(
                input_ids, past_key_values=past_key_values, output_attentions=True
            )
            attentions_ltm = torch.stack([tensor.cpu() for tensor in outputs_ltm.attentions]).squeeze(1)
            last_hidden_states = outputs_ltm.last_hidden_state

            new_token = model.base_model.lm_head(last_hidden_states)[0, -1].argmax()
            input_ids = torch.cat((input_ids[:, 1:], new_token.view(1, 1)), dim=1)

            outputs_ssm = model.ea_layer(last_hidden_states, input_ids, output_attentions=True)
            attentions_ssm = torch.stack([tensor.cpu() for tensor in outputs_ssm[1]]).squeeze(1)

            outputs_ltm = model.base_model.model(
                new_token.view(1, 1), past_key_values=past_key_values, output_attentions=True
            )
            attentions_new = torch.stack([tensor.cpu() for tensor in outputs_ltm.attentions]).squeeze(1)

            score_ltm = attentions_new.squeeze(2)
            score_h2o = attentions_ltm.sum(dim=2)
            score_ssm = attentions_ssm[:, :, -1, :].mean(dim=(0, 1))
            
            for layer in range(score_ltm.size(0)):
                tmp1 = 0
                tmp2 = 0
                for head in range(score_ltm.size(1)):
                    tmp1 += sim(score_ltm[layer, head, 1:-50], score_ssm[:-50], k=49).item() + score_ltm[layer, head, 0].item()
                    tmp2 += sim(score_ltm[layer, head, :-50], score_h2o[layer, head, :-50], k=50).item()
                    # tmp1 += cosine_sim(score_ltm[layer, head, 1:], score_ssm).item()
                    # tmp2 += cosine_sim(score_ltm[layer, head, :-1], score_h2o[layer, head]).item()
                results_ssm[layer] += tmp1 / score_ltm.size(1) / len(prompts)
                results_h2o[layer] += tmp2 / score_ltm.size(1) / len(prompts)
    
    plt.figure(figsize=(10, 6))
    layers = list(range(len(results_ssm)))
    plt.plot(layers, results_ssm, marker='o', linestyle='-', linewidth=2, markersize=6, label="SSM")
    plt.plot(layers, results_h2o, marker='s', linestyle='-', linewidth=2, markersize=6, label="H2O")
    plt.xlabel("Layer Number", fontsize=14)
    plt.ylabel("Sum of Attention Scores for Selected Tokens", fontsize=14)
    plt.title("Attention Score Comparison by Layer", fontsize=16)
    plt.xticks(layers)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{model_name}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU 번호를 CLI 인자로 받아 실행하는 모델 스크립트")
    parser.add_argument("--gpu", type=int, default=0, help="사용할 GPU 번호 (기본값: 0)")
    args = parser.parse_args()
    main(args)
