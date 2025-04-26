import torch
from transformers import AutoTokenizer
from modelscope.models import Model

# 加载模型和分词器
model_dir = "/data/csyData/models/iic/nlp_corom_sentence-embedding_chinese-base-medical"
model = Model.from_pretrained(model_dir, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

def batch_encode(texts):
    """批量编码函数（核心改进点）"""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
        return_attention_mask=True
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"]
    }

# 示例批量输入
batch_queries = [
    "上消化道出血手术大约多久",
    "糖尿病患者的常规护理措施",
    "心肌梗塞的急救处理流程"
]

# 批量编码
batch_inputs = batch_encode(batch_queries)

# GPU加速（可选）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

# 批量推理
with torch.no_grad():
    batch_embeddings = model.encode(**batch_inputs)  # 形状 [batch_size, hidden_size]

print(f"批量嵌入维度：{batch_embeddings.shape}")  # 应输出 torch.Size([3, 768])