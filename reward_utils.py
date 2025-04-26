import re
from typing import *
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
from transformers import AutoTokenizer
from modelscope.models import Model


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    # q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# ----------------------- 上面的是GRPO示例给的奖励函数，没有采用  ----------------------------------



def greedy_linear_sum_assignment(sim_matrix):
    sim_matrix = np.array(sim_matrix)
    row_count, col_count = sim_matrix.shape
    assigned_rows = set()
    assigned_cols = set()
    matches = []

    flat_indices = [(i, j, sim_matrix[i, j]) 
                    for i in range(row_count) 
                    for j in range(col_count)]
    flat_indices.sort(key=lambda x: -x[2])

    for i, j, score in flat_indices:
        if i not in assigned_rows and j not in assigned_cols:
            matches.append((i, j))
            assigned_rows.add(i)
            assigned_cols.add(j)

    return zip(*matches)



def extract_entities(generated: str) -> List[Dict]:
    """从乱序文本中提取结构化信息"""
    V3result = {}
    pattern = r'\*\*(.*?)\*\*'
    key = None
    for text in generated.splitlines(keepends=True):
        if bool(re.search(r'## ', text)) and bool(re.search(r'\*\*', text)):
            matches = re.findall(pattern, text, re.DOTALL)
            if len(matches) >= 2:
                cleaned_matches = []
                for match in matches:
                    cleaned_match = re.sub(r'^\d+(?:-\d+)?\.\s*', '', match.strip())
                    cleaned_matches.append(cleaned_match)
                key = "&".join(cleaned_matches)
                V3result[key] = ""
            elif len(matches) == 1:
                key = re.sub(r'^\d+(?:-\d+)?\.\s*', '', matches[0].strip())
                V3result[key] = ""
        elif key:
            V3result[key] += text
    return V3result


def get_reward_fn(model=None,tokenizer=None):
    def compute_rewards(completions, answer, **reward_kwargs) -> Dict:
        """计算奖励"""
        all_score = []
        for ans, generated in zip(answer,completions):
            generated = generated[0]['content']
            ground_truth = ans
            expected_diseases = list(ground_truth.keys())
            
            parseddict = extract_entities(generated)
            pred_diseases = list(parseddict.keys())
            total_reward = 0

            # 1. 基础格式奖励
            if re.search(r"## \*\*.+?\*\*：", generated):
                total_reward += 0.3  
            if re.findall(r"\n-\ \*\*.+?\*\*：", generated):
                total_reward += 0.2

            def batch_encode(texts):
                encoded = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                    return_attention_mask=True
                )
                batch_inputs = {
                                k: v.to(model.device) for k, v in {
                                    "input_ids": encoded["input_ids"],
                                    "attention_mask": encoded["attention_mask"]
                                }.items()
                            }
                with torch.no_grad():
                    batch_embeddings = model.encode(**batch_inputs)
                return batch_embeddings

            # 2：疾病识别准确率
            if len(pred_diseases) > 0:
                gold_embeds = batch_encode(expected_diseases)
                pred_embeds = batch_encode(pred_diseases)

                # 相似度矩阵，这个是计算预测的疾病和实际的疾病名称的emb的相似度的矩阵，
                sim_matrix = util.pytorch_cos_sim(gold_embeds, pred_embeds).cpu().numpy()

                # 匹配矩阵，因为输出的疾病和实际的疾病顺序可能不一样，比如模型预测出了疾病1,2,3，但是标准答案是疾病3,2,1，你也不能说模型输出错误了
                row_ind, col_ind = greedy_linear_sum_assignment(sim_matrix)

                matched = []
                reason_scores = []
                for i, j in zip(row_ind, col_ind):
                    sim = sim_matrix[i][j]
                    if sim >= 0.6:   # 看疾病的embedding相似度如果大于0.6
                        gold_disease = expected_diseases[i]
                        pred_disease = pred_diseases[j]
                        gold_reason = ground_truth[gold_disease]
                        pred_reason = parseddict[pred_disease]

                        # 计算诊断来源的emb的相似度
                        reason_embed_gold = batch_encode(gold_reason)
                        reason_embed_pred = batch_encode(pred_reason)
                        sim_score = util.pytorch_cos_sim(reason_embed_gold, reason_embed_pred).item()
                        reason_scores.append(sim_score)
                        matched.append((gold_disease, pred_disease, round(sim, 3), round(sim_score, 3)))

                recall = len(matched) / len(expected_diseases) if expected_diseases else 0   # 诊断疾病的得分
                avg_reason_score = np.mean(reason_scores) if reason_scores else 0  # 诊断理由的得分

                total_reward += recall
                total_reward += avg_reason_score
            all_score.append(total_reward)
        return all_score
    return compute_rewards

if __name__ == "__main__":
    reference_diseases = {
        "支气管哮喘": "主诉",
        "慢性支气管炎": "主诉",
        "慢性胃炎": "既往史",
        "反流性食管炎": "既往史"
    }


    model_dir = "/data/csyData/models/iic/nlp_corom_sentence-embedding_chinese-base-medical"
    model = Model.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # 测试案例：乱序输出+格式干扰
    test_output = """
    ## **慢性胃炎**: 这里有个格式错误示范
    ## **反流性食管炎**:
      - **既往史**: 确诊5年，长期服用PPI
    ## **支气管哮喘** 
     - **主诉**：活动后喘息（缺少冒号）
    ## **不存在的疾病**:
      - **既往史**: 测试异常数据
    """
    rewards_fn = get_reward_fn(model,tokenizer)
    result = rewards_fn([reference_diseases], [test_output])
    print(f"""
    疾病覆盖率：{result[0]:.2f}（应得0.75）
    来源准确率：{result[1]:.2f}（应得0.5）
    """)