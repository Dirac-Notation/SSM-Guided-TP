import torch
import json
import math
import os
import argparse

from tqdm import tqdm
from rouge_score import rouge_scorer
from transformers import AutoTokenizer

from eagle.model.ea_model import EaModel

from eagle.model.cnets_modify import Model_modify

def args_budget(value):
    try:
        return int(value)
    except:
        return None

parser = argparse.ArgumentParser(description="EAGLE 모델 성능 측정")
parser.add_argument("--ltm_path", type=str, help="LTM 모델", default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--ssm_path", type=str, help="SSM 모델", default="yuhuili/EAGLE-llama2-chat-7B") # trained_model/two_layer / trained_model/no_feature
parser.add_argument("--dataset", type=str, help="데이터셋", default="fewshot_data/cnn_dailymail-3shot.jsonl")
parser.add_argument("--method", type=str, help="토큰 가지치기 기법", default="full")
parser.add_argument("--token_budget", type=args_budget, nargs="+", help="토큰 버짓", default=[None])
parser.add_argument("--gpu", type=int, help="사용할 GPU 넘버", default=0)

args = parser.parse_args()

base_model_path = args.ltm_path
EAGLE_model_path = args.ssm_path
dataset = args.dataset
method = args.method
token_budgets = args.token_budget
gpu = args.gpu

rouge_types = ["rouge1", "rouge2", "rougeL"]

# 프롬프트 전처리
with open(dataset, "r") as f:
    prompts = []
    num_prompts = []
    answers = []
    num_answers = []
    
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
        num_prompts.append(input_ids.size(1))
        answers.append(answer)
        num_answers.append(tokenizer(answer, return_tensors="pt").input_ids.size(1))
    
    del tokenizer

datalen = len(prompts)
prompt_length = math.ceil(sum(num_prompts)/len(num_prompts))

scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)

model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    torch_dtype=torch.float16,
    device_map=f"cuda:{gpu}",
).eval()

for token_budget in token_budgets:
    model.set_token_budget(method, token_budget)
    
    for i in tqdm(range(10), desc="Warm Up"):
        model.eagenerate(prompts[i].to(model.base_model.device), max_new_tokens=512)

    init_time = 0
    decode_time = 0
    ltm_time = 0
    ssm_time = 0
    total_round = 0

    total_acceptance_ratio = 0
    total_acceptance_length = 0

    total_rouge1 = 0
    total_rouge2 = 0
    total_rougeL = 0

    dataset_name = dataset.replace("data/", "").replace(".jsonl", "")
    os.makedirs("output_text", exist_ok=True)
    with open(f"output_text/{token_budget if token_budget is not None else 'Full'}.jsonl", "w") as file:
        for input_ids, num_ids, answer, num_answer in tqdm(zip(prompts, num_prompts, answers, num_answers), total=len(prompts), desc="Running Test"):
            
            input_ids = input_ids.to(model.base_model.device)

            output_ids, length, num_round, acceptance_ratio, acceptance_length, decoding_time = model.eagenerate(input_ids, max_new_tokens=128)
            
            init_time += decoding_time[0]
            decode_time += decoding_time[1]
            ltm_time += decoding_time[2]
            ssm_time += decoding_time[3]
            total_round += num_round
            total_acceptance_ratio += acceptance_ratio
            total_acceptance_length += acceptance_length

            # Decode the output_ids into a string
            generated_text = model.tokenizer.decode(output_ids[0][num_ids:], skip_special_tokens=True)

            json.dump({"output": generated_text}, file)
            file.write("\n")
            
            # Calculate ROUGE scores
            rouge_scores = scorer.score(answer, generated_text)
            
            total_rouge1 += rouge_scores["rouge1"].fmeasure
            total_rouge2 += rouge_scores["rouge2"].fmeasure
            total_rougeL += rouge_scores["rougeL"].fmeasure

    print(f"dataset: {dataset}")
    print(f"average prompt length: {prompt_length}, Min prompt length: {min(num_prompts)}, Max prompt length: {max(num_prompts)}")
    
    print(f"method: {method}")
    print(f"token budget: {token_budget if token_budget is not None else 'Full'}")

    print(f"average init throughput = {1e6*init_time/datalen}")
    print(f"average decode throughput = {1e6*decode_time/datalen}")
    print(f"average ltm latency = {ltm_time/datalen/1e3}")
    print(f"average ssm latency = {ssm_time/datalen/1e3}")
    print(f"average round = {total_round/datalen}")

    print(f"average acceptance ratio = {total_acceptance_ratio/datalen}")
    print(f"average acceptance length = {total_acceptance_length/datalen}")

    print(f"average ROUGE-1 = {total_rouge1/datalen}")
    print(f"average ROUGE-2 = {total_rouge2/datalen}")
    print(f"average ROUGE-L = {total_rougeL/datalen}")