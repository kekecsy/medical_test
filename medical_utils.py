from datasets import Dataset
import json
from transformers import AutoTokenizer
from datasets import load_dataset
import random
from datasets import Dataset
SYSTEM_PROMPT = """
以下是描述任务的指令，以及提供更多上下文的输入。
请写出恰当完成该请求的回答。
在回答之前，请仔细思考问题，并创建一个逐步的思维链，以确保回答合乎逻辑且准确。
### Instruction:  你是一位在临床推理、诊断和治疗计划方面具有专业知识的医学专家。
请回答以下医学问题。
### Question:  {}
### Response:  <think>{}
"""

SYSTEM_PROMPT2 = """
你是一个专业的医疗助手，请根据以下患者入院记录，整理出规范的诊断列表，并说明移除某些诊断的原因：


要求：
1. 列出所有核心诊断（需有明确证据支持）
2. 移除推测性或治疗过程类描述
3. 对每个移除的诊断说明原因

请按以下格式输出：
1. **[诊断1]**  
- 支持依据：   

2. **[诊断2]**  
- 支持依据：

...

### 移除诊断及其原因

1. **[排除诊断1]**  
- 理由：   

...

"""

SYSTEM_PROMPT_HARD = """
作为住院医师，请根据以下信息生成入院诊断分析：

【分析要求】
分三步处理：
1. 关键信息提取
   - 用<症状><体征><检查>标记核心临床证据
   - 标注手术史与当前症状的时间关联性

2. 诊断推理
   - 给出每一项诊断的可能原因
   - 每个诊断必须包含：
     √ 支持点：至少1个症状+1个体征/检查
     √ 病理机制：用箭头连接证据与病理过程（如胸痛→心肌缺血→ST段抬高）

3. 鉴别排除
   - 列出最需区分的1-2个疾病
   - 说明排除依据：缺失的关键症状/矛盾检查结果

【输出格式】
### 诊断分析 ###
主诉线索：<...>
核心体征：<...>
检查提示：<...>

诊断：
1. [疾病名称]
   - 支持证据：症状A + 检查B
   - 病理过程：症状→机制→检查表现

需排除疾病：
- [疾病名称]：缺乏[关键指标]/[检查C]显示[矛盾结果]
"""


record_prompt = """
### 入院记录：
{}

"""

answer_prompt = """

### 诊断结果
{}

"""

def get_questions(data_path, tokenizer) -> Dataset:
    with open(data_path) as f:
        data = json.load(f)
    result = {'prompt':[],'answer':[]}
    for item in data:
        description = item["description"]
        qa_pairs = item["QA_pairs"]
        for qa in qa_pairs:
            question = qa["question"]
            answer = qa["answer"]
            
            # 组合 description 和 QA 对
            result['prompt'].append(
                [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': description+question}
                ]
            )
            result['answer'].append(answer)
    return Dataset.from_dict(result)


def get_questions2(data_path, tokenizer) -> Dataset:
    result = {'text':[]}
    with open(data_path) as f:
        for line in f:
            data_i = json.loads(line)
            for item in data_i:
                result['text'].append(tokenizer.apply_chat_template([
                    {'role': 'system', 'content': SYSTEM_PROMPT2},
                    {'role': 'user', 'content': record_prompt.format(data_i[item]["simple_adm_record"])},
                    {'role':'assistant','content': answer_prompt.format(data_i[item]["simple_adm_judge"])}
                    ], 
                    tokenize=False
                ))
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            max_length=4096,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
    dataset = Dataset.from_dict(result).map(
        tokenize_fn,
        batched=True,
        batch_size=32,
        num_proc=4,
        remove_columns=["text"]  # 清理中间文本列
    )
    return dataset


def get_questions2_hard(data_path, tokenizer) -> Dataset:
    result = {'text':[]}
    with open(data_path) as f:
        for line in f:
            data_i = json.loads(line)
            for item in data_i:
                result['text'].append(tokenizer.apply_chat_template([
                    {'role': 'system', 'content': SYSTEM_PROMPT_HARD},
                    {'role': 'user', 'content': record_prompt.format({k:v for k,v in data_i[item]["admission_record"].items() if k != '入院诊断'})},
                    {'role':'assistant','content': answer_prompt.format(data_i[item]["adm_response"])}
                    ], 
                    tokenize=False
                ))
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            max_length=4096,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
    dataset = Dataset.from_dict(result).map(
        tokenize_fn,
        batched=True,
        batch_size=32,
        num_proc=4,
        remove_columns=["text"]  # 清理中间文本列
    )
    return dataset


def create_formatting_record_func(tokenizer):
    def formatting_prompts_func(example):
        text = tokenizer.apply_chat_template([
                    {'role': 'system', 'content': '你是一位经验丰富的临床医生，致力于从入院病历中给出初步诊断'},
                    {'role': 'user', 'content': example['instruction']},
                    {'role':'assistant','content': example["output"]}
                    ],
                    tokenize=False
                )
        return text
    return formatting_prompts_func


def formatting_prompts_func(example):
    text = f"### Question: {example['instruction']}\n ### Answer: {example['output']}"
    return text

def create_formatting_func(tokenizer):
    def formatting_prompts_func(example):
        text = tokenizer.apply_chat_template([
                    {'role': 'system', 'content': '你是一位经验丰富的呼吸科医生，致力于为患者提供准确、专业的呼吸疾病诊断与治疗建议。'},
                    {'role': 'user', 'content': example['instruction']},
                    {'role':'assistant','content': example['output']}
                    ], 
                    tokenize=False
                )
        return text
    return formatting_prompts_func

def create_test_func(tokenizer):
    def formatting_prompts_func(example):
        text = tokenizer.apply_chat_template([
                    {'role': 'user', 'content': example['instruction']}
                    ], 
                    tokenize=False
                )
        return text
    return formatting_prompts_func


def get_mokeqa(data_path, tokenizer) -> Dataset:
    # result = {'text':[]}
    # processed_data = []
    # with open(data_path) as f:
    #     data = json.load(f)

    # 方案2：
    
    return load_dataset("json",data_files=data_path, split="train")

    # 方案1
    #     max_tokenlen = 0
    #     large_seq = []
    #     for item in data:
    #         text = tokenizer.apply_chat_template([
    #             {'role': 'system', 'content': item['system']},
    #             {'role': 'user', 'content': item['instruction']},
    #             {'role':'assistant','content': item['output']}
    #             ], 
    #             tokenize=False
    #         )
    #         result['text'].append(text)
    #         if len(tokenizer.encode(text)) > max_tokenlen:
    #             max_tokenlen = len(tokenizer.encode(text))
    #             if len(tokenizer.encode(text)) > 1024:
    #                 large_seq.append(text)
    # def tokenize_fn(examples):
    #     return tokenizer(
    #         examples["text"],
    #         max_length=1536,
    #         truncation=True,
    #         padding="max_length",
    #         return_tensors="pt"
    #     )
    # dataset = Dataset.from_dict(result).map(
    #     tokenize_fn,
    #     batched=True,
    #     batch_size=32,
    #     num_proc=4,
    #     remove_columns=["text"]  # 清理中间文本列
    # )
    # return dataset



class AdmDataset(Dataset):
    def __init__(self, data_path):
        """
        初始化数据集。
        :param data_path: 数据文件路径
        """
        self._data = self._load_data(data_path)

    def _load_data(self, data_path):
        """
        加载原始 JSON 数据。
        :param data_path: 数据文件路径
        :return: 原始数据列表
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        return raw_data

    def __len__(self) -> int:
        """
        返回数据集的长度。
        """
        return len(self.raw_data)

    def __getitem__(self, idx):
        """
        根据索引获取数据，并在获取时处理 output 拼接。
        :param idx: 索引
        :return: 数据项
        """
        item = self._data[idx]

        randomized_output = list(item['output'].items())
        random.shuffle(randomized_output)  # 随机打乱顺序
        output_text = "\n".join(
            [f"## **{key}**: {value}" for key, value in randomized_output]
        )

        processed_item = {
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": output_text,
            "system": item.get("system", "")
        }

        return processed_item



if __name__ == '__main__':
    # data_path = '/data/csydata/deepseek_test/datasets/medical/medical_mokeqa.json'
    # model_path = "/data/csydata/deepseek_test/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # dataset = get_mokeqa(data_path,tokenizer)
    data_path = "/data/csydata/deepseek_test/datasets/medical/medical_clean1.json"
    dataset = AdmDataset(data_path)

    print(f"数据集大小: {len(dataset)}")