from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score
from scipy.optimize import linear_sum_assignment
import numpy as np
from FlagEmbedding import BGEM3FlagModel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# disease_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
disease_encoder = BGEM3FlagModel('/data/csyData/models/BAAI/bge-base-zh-v1.5',  use_fp16=True)


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


def evaluate_predictions(gold: Dict[str, str], pred: Dict[str, str], threshold=0.75):
    gold_diseases = list(gold.keys())
    pred_diseases = list(pred.keys())

    # 计算疾病名称的句向量
    gold_embeds = disease_encoder.encode(gold_diseases, num_workers=0)['dense_vecs']
    pred_embeds = disease_encoder.encode(pred_diseases, num_workers=0)['dense_vecs']

    # 相似度矩阵
    sim_matrix = util.pytorch_cos_sim(gold_embeds, pred_embeds).cpu().numpy()

    # 匹配
    row_ind, col_ind = greedy_linear_sum_assignment(sim_matrix)

    matched = []
    reason_scores = []
    for i, j in zip(row_ind, col_ind):
        sim = sim_matrix[i][j]
        if sim >= threshold:
            gold_disease = gold_diseases[i]
            pred_disease = pred_diseases[j]
            gold_reason = gold[gold_disease]
            pred_reason = pred[pred_disease]

            # BERTScore（中文推理句子）
            reason_embed_gold = disease_encoder.encode(gold_reason, num_workers=0)['dense_vecs']
            reason_embed_pred = disease_encoder.encode(pred_reason, num_workers=0)['dense_vecs']
            sim_score = util.pytorch_cos_sim(reason_embed_gold, reason_embed_pred).item()
            reason_scores.append(sim_score)
            matched.append((gold_disease, pred_disease, round(sim, 3), round(sim_score, 3)))

    recall = len(matched) / len(gold_diseases) if gold_diseases else 0
    avg_reason_score = np.mean(reason_scores) if reason_scores else 0

    print("【匹配结果】：")
    for real, pred, name_sim, reason_sim in matched:
        print(f"真实: {real}  预测: {pred} | 疾病相似度: {name_sim} | 推理相似度: {reason_sim}")

    print("\n【评估指标】：")
    print(f"匹配数量: {len(matched)} / {len(gold_diseases)}")
    print(f"召回率 Recall: {round(recall, 3)}")
    print(f"平均推理相似度: {round(avg_reason_score, 3)}")


def main():
    # 模拟真实病历
    gold = {
        "肺炎": "患者咳嗽发热，肺部听诊有湿罗音",
        "慢性阻塞性肺病": "有吸烟史，呼吸困难，肺功能下降",
        "支气管哮喘": "夜间喘息发作，吸入激素有效"
    }

    # 模拟模型预测结果
    pred = {
        "呼吸道感染": "体温升高，咳嗽、咽痛，考虑感染",
        "慢阻肺": "病人有呼吸困难，肺功能受损",
        "哮喘": "夜间咳嗽喘息，用药后缓解"
    }

    evaluate_predictions(gold, pred)


if __name__ == '__main__':
    main()