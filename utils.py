import json
import torch

from tqdm import tqdm

def load_datasets(
    dataset_path: str,
    tokenizer
):
    with open(dataset_path, "r") as f:
        prompts = []
        answers = []
        
        datalines = f.readlines()
        articles = []
        
        for dataline in datalines:
            articles.append(json.loads(dataline))
        
    for data in tqdm(articles, desc="Tokenizing"):
        input = data["article"]
        answer = data["summary_gt"]
        
        input_ids = tokenizer(input, return_tensors="pt").input_ids

        prompts.append(input_ids)
        answers.append(answer)
    
    return prompts, answers

def sim(
    vec_1: torch.Tensor,
    vec_2: torch.Tensor,
    k: int = 20,
):
    if vec_1.dim() != 1 or vec_2.dim() != 1:
        assert "must dim 1"

    index_set = vec_2.topk(k).indices

    return (vec_1.index_select(dim=0, index=index_set)).sum()

def set_sim(
    vec_1: torch.Tensor,
    vec_2: torch.Tensor,
    k: int = 20,
):
    if vec_1.dim() != 1 or vec_2.dim() != 1:
        assert "must dim 1"

    # top-k 인덱스 추출
    idx1 = torch.topk(vec_1, k, largest=True).indices
    idx2 = torch.topk(vec_2, k, largest=True).indices

    # Python set으로 변환
    set1 = set(idx1.tolist())
    set2 = set(idx2.tolist())

    # 교집합, 합집합 계산
    inter = set1 & set2
    uni   = set1 | set2

    # 합집합이 비어있으면 0.0 반환
    if len(uni) == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=vec_1.device)

    # Jaccard similarity
    score = len(inter) / len(uni)
    return torch.tensor(score, dtype=torch.float32, device=vec_1.device)

def cosine_sim(
    vec_1: torch.Tensor,
    vec_2: torch.Tensor,
):
    if vec_1.dim() != 1 or vec_2.dim() != 1:
        assert "must dim 1"

    dot = torch.dot(vec_1, vec_2)
    norm1 = torch.norm(vec_1)
    norm2 = torch.norm(vec_2)
    return dot / (norm1 * norm2 + 1e-8)

def diff(
    vec_1: torch.Tensor,
    vec_2: torch.Tensor,
    vec_3: torch.Tensor,
    k: int = 128
):
    if vec_1.dim() != 1 or vec_2.dim() != 1 or vec_3.dim() != 1:
        assert "must dim 1"
    
    set_1 = set(vec_1.topk(k).indices.tolist())
    set_2 = set(vec_2.topk(k).indices.tolist())
    set_3 = set(vec_3.topk(25).indices.tolist())
    
    union_all = set_1&set_2&set_3
    union_12 = set_1&set_2 - union_all
    union_13 = set_1&set_3 - union_all
    union_23 = set_2&set_3 - union_all

    num_union_all = len(union_all)
    num_union_12 = len(union_12)
    num_union_13 = len(union_13)
    num_union_23 = len(union_23)

    try:
        score_union_all = vec_1[torch.tensor(list(union_all))].sum().item()
    except:
        score_union_all = 0

    try:
        score_union_12 = vec_1[torch.tensor(list(union_12))].sum().item()
    except:
        score_union_12 = 0

    try:
        score_union_13 = vec_1[torch.tensor(list(union_13))].sum().item()
    except:
        score_union_13 = 0

    return num_union_12, num_union_13, num_union_23, num_union_all, score_union_12, score_union_13, score_union_all, vec_1[torch.tensor(list(set_1))].sum().item()