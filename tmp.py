import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import load_datasets, sim

# 모델과 토크나이저 불러오기
model_name = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompts, answers = load_datasets("fewshot_data/cnn_dailymail-0shot.jsonl", tokenizer)
print(sum(t.numel() for t in prompts)/len(prompts))

# 텍스트 입력
prompt = prompts[0]
inputs = prompt.to(model.device)

# 모델 추론
with torch.inference_mode():
    outputs = model(inputs, output_attentions=True)

print(sim(outputs.attentions[0][0,0,-1], outputs.attentions[0][0,0,-2], 100))