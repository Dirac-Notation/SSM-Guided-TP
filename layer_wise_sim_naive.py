import os
import torch
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import sim, cosine_sim, set_sim, load_datasets  # prompts 로드용 유틸

def main(gpu: int):
    device = f"cuda:{gpu}"

    # 고정 모델 이름
    small_model_name = "meta-llama/Llama-3.2-1B" # "facebook/opt-350m"
    large_model_name = "meta-llama/Llama-3.1-8B" # "facebook/opt-6.7b"
    dataset_path     = "fewshot_data/cnn_dailymail-0shot.jsonl"

    # 토크나이저 및 모델 로드
    tokenizer   = AutoTokenizer.from_pretrained(small_model_name)
    small_model = AutoModelForCausalLM.from_pretrained(small_model_name).to(torch.bfloat16).to(device).eval()
    large_model = AutoModelForCausalLM.from_pretrained(large_model_name).to(torch.bfloat16).to(device).eval()

    # 프롬프트만 사용
    prompts, _ = load_datasets(dataset_path, tokenizer)

    num_layers_ssm = small_model.config.num_hidden_layers  # 12
    num_layers_ltm = large_model.config.num_hidden_layers  # 32
    num_heads_ltm  = large_model.config.num_attention_heads # 32

    # [layer, head] 유사도 합 저장
    results_ssm = torch.zeros((num_layers_ssm, num_layers_ltm, num_heads_ltm), device='cpu')
    results_h2o = torch.zeros((num_layers_ltm, num_heads_ltm), device='cpu')

    for prompt in tqdm(prompts, desc="Computing similarities"):
        input_ids = prompt.to(device)
        with torch.inference_mode():
            out_small = small_model(input_ids, output_attentions=True)
            out_large = large_model(input_ids, output_attentions=True)

        attn_small = torch.stack(out_small.attentions).cpu().squeeze(1)  # [12, heads, seq, seq]
        attn_large = torch.stack(out_large.attentions).cpu().squeeze(1)  # [32, heads, seq, seq]

        score_ssm = attn_small[:,:,-1,:].mean(dim=1) # [12, seq]
        score_h2o = attn_large[:,:,:-1,:].sum(dim=2) # [32, heads, seq]
        
        for ltm_layer in range(num_layers_ltm):
            for head in range(num_heads_ltm):
                a = attn_large[ltm_layer, head, -1, :-51]
                
                b = score_h2o[ltm_layer, head, :-51]
                # results_h2o[ltm_layer, head] += set_sim(a, b, k=50)
                results_h2o[ltm_layer, head] += cosine_sim(a, b)

        for ssm_layer in range(num_layers_ssm):
            for ltm_layer in range(num_layers_ltm):
                for head in range(num_heads_ltm):
                    a = attn_large[ltm_layer, head, -1, :-51]
                    
                    b = score_ssm[ssm_layer, :-51]
                    # results_ssm[ssm_layer, ltm_layer, head] += set_sim(a, b, k=50)
                    results_ssm[ssm_layer, ltm_layer, head] += cosine_sim(a, b)

    # 평균 유사도 계산
    results_ssm /= len(prompts)
    results_h2o /= len(prompts)
    
    results_ssm = results_ssm.mean(dim=2)  # [12, 32]
    results_h2o = results_h2o.mean(dim=1) # [32]
    
    # 3x4 서브플롯: 레이어별 헤드 유사도 라인 차트 (개선판)
    row = 4
    column = 4
    fig, axes = plt.subplots(row, column, figsize=(4*column, 4*row), sharex=True, sharey=True)
    num_heads = results_ssm.size(1)
    x = list(range(num_heads))

    for layer in range(num_layers_ssm):
        ax = axes[layer // column, layer % column]
        # SSM, H2O 선 그리기 (마커/라인 두께 설정)
        ax.plot(x, results_ssm[layer].tolist(),
                marker='o', linewidth=1.5, label="SSM")
        ax.plot(x, results_h2o.tolist(),
                marker='s', linewidth=1.5, label="H2O")

        # 서브플롯 제목 및 축 레이블
        ax.set_title(f"Layer {layer}", fontsize=12, pad=6)
        ax.set_xlabel("Layer", fontsize=10)
        if layer % 4 == 0:
            ax.set_ylabel("Top-k Similarity", fontsize=10)

        # x축 티크, y축 그리드만 표시
        ax.set_xticks(x)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 전역 제목, 전역 레전드 추가
    fig.suptitle("Attention Similarity per Layer & Head", fontsize=16, y=1.02)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12, frameon=False)

    # 레이아웃 조정 및 저장
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/llama3.1-8Bvsllama3.2-1B_similarity.png", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OPT-125m과 OPT-6.7b의 Attention 유사도 비교 스크립트"
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="사용할 GPU 번호 (기본값: 0)"
    )
    args = parser.parse_args()
    main(args.gpu)
