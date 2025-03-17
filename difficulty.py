import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

from eagle.model.cnets import Model
from eagle.model.configs import EConfig

def js_divergence(P, Q):
    M = (P + Q)/2
    return 0.5 * (kl_divergence(P, M) + kl_divergence(Q, M))

def kl_divergence(P, Q):
    epsilon = 1e-10
    P = P + epsilon
    Q = Q + epsilon
    return torch.sum(P * torch.log(P/Q))

def du_distance(vec1, vec2):
    return torch.sqrt(torch.sum((vec1-vec2)**2)).item()

device = "cuda:2"

# 모델과 토크나이저 로드
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).half().to(device=device)

config = EConfig.from_pretrained("trained_model/original/config.json")
draft_model = Model(config)
# draft_model.load_state_dict(torch.load(hf_hub_download("yuhuili/EAGLE-llama2-chat-7B", filename="pytorch_model.bin")))
draft_model.load_state_dict(torch.load("trained_model/original/pytorch_model.bin"))
draft_model.half().to(device=device)

# 모델을 평가 모드로 설정
model.eval()
draft_model.eval()

# 초기 입력 (빈 문자열 또는 원하는 텍스트)
with open("fewshot_data/cnn-1shot.jsonl", "r") as f:
    files = f.readlines()

dist = 0

for idx in range(5):
    data = json.loads(files[idx])
    input_text = data["article"]

    # 입력을 토큰화
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # 생성된 토큰과 확률을 저장할 리스트
    tokens = []
    probabilities = []
    colors = []
    draft_probabilities = []
    draft_colors = []
    
    similarities = []
    distances = []

    # 토큰 생성 반복
    for seq_id in range(32):
        # forward 패스를 통해 다음 토큰 예측
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            
            hidden_states = outputs.hidden_states[-1]
            logits = outputs.logits
            
            draft_hidden_states = draft_model(hidden_states[:,:-1], input_ids[:,1:])
            draft_logits = model.lm_head(draft_hidden_states).to(logits.dtype)

        # LTM 확률 및 토큰 계산
        next_token_logits = logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        next_token_id = torch.argmax(next_token_probs, dim=-1)
        next_token_prob = next_token_probs[0, next_token_id].item()
        
        # SSM 확률 및 토큰 계산
        draft_token_logits = draft_logits[:, -1, :]
        draft_token_probs = torch.softmax(draft_token_logits, dim=-1)
        draft_token_prob, draft_token_id = torch.topk(draft_token_probs, k=3, dim=-1)

        # print(f"Next token: {next_token_id}, Token prob: {next_token_prob}")
        # print(f"Draft token: {draft_token_id}, Token prob: {draft_token_prob}")
        # print((draft_token_id == next_token_id.item()).sum())
        # print()

        # 생성된 토큰과 그 확률을 리스트에 저장
        next_token = tokenizer.decode(next_token_id)
        tokens.append(repr(next_token))
        probabilities.append(next_token_prob)
        draft_probabilities.append(draft_token_probs[0, next_token_id.item()].item())
        
        if (draft_token_id == next_token_id.item()).sum():
            colors.append("tab:blue")
            draft_colors.append("tab:cyan")
        else:
            colors.append("tab:red")
            draft_colors.append("tab:pink")

        # 생성된 토큰을 입력 시퀀스에 추가
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
        
        last_hidden_states = hidden_states[0,-1,:]
        last_draft_hidden_states = draft_hidden_states[0,-1,:]
        # similarity_states = (torch.dot(last_hidden_states, last_draft_hidden_states)/(last_hidden_states.norm(dim=-1)*last_draft_hidden_states.norm(dim=-1))).item()
        similarity_states = du_distance(last_hidden_states, last_draft_hidden_states)
        similarity_logits = (torch.dot(logits[0,-1,:], draft_logits[0,-1,:])/(logits[0,-1,:].norm()*draft_logits[0,-1,:].norm())).item()
        divergence = js_divergence(next_token_probs, draft_token_probs).item()
        
        similarities.append(similarity_states)
        distances.append(divergence)
        
        os.makedirs(f"difficulty/distribution/{idx}", exist_ok=True)
        plt.title(f"Token: {repr(next_token)}\nLHS Distance: {similarity_states:.2f} | Prob Divergence: {divergence:.2f}", fontsize=15)
        plt.plot(next_token_probs.squeeze().cpu(), alpha = 0.7, label="LTM")
        plt.plot(draft_token_probs.squeeze().cpu(), alpha = 0.7, label="EAGLE")
        plt.xlabel("Token ID")
        plt.ylabel("Probability")
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"difficulty/distribution/{idx}/{seq_id}.png")
        plt.close()
    
    os.makedirs("difficulty/dist_sim", exist_ok=True)
    plt.scatter(similarities, distances)
    plt.xlabel("Last Hidden State Similarity")
    plt.ylabel("Probability Distance")
    plt.tight_layout()
    plt.savefig(f"difficulty/dist_sim/{idx}.png")
    plt.close()

    os.makedirs("difficulty/original", exist_ok=True)
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(tokens))-width/2, probabilities, width=width, color=colors, label="LTM")
    plt.bar(np.arange(len(tokens))+width/2, draft_probabilities, width=width, color=draft_colors, label="EAGLE")
    plt.xticks(np.arange(len(tokens)), tokens, rotation=90)
    plt.xlabel('Generated Token')
    plt.ylabel('Probability')
    plt.title('Generated Tokens and Their Probabilities')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"difficulty/original/{idx}.png")
    plt.close()
    
    dist += sum(distances)

print(dist/160)