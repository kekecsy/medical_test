# grpo_train.py

import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
os.environ["MASTER_PORT"] = "29501"

import json
import torch
from os.path import exists
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer, DataCollatorForCompletionOnlyLM
from transformers import (TrainingArguments, 
                        AutoModelForCausalLM, 
                        AutoTokenizer, 
                        BitsAndBytesConfig, 
                        default_data_collator)
import sglang as sgl
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import Accelerator
from medical_utils import get_mokeqa, create_formatting_record_func, AdmDataset
from FlagEmbedding import BGEM3FlagModel
from reward_utils import get_reward_fn

from modelscope.models import Model

def process_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for i in data:
        i['output'] = "\n".join([f"## **{key}**：\n {value}" for key, value in i['output'].items()])
    dataset = Dataset.from_list(data)
    
    processed_dataset = dataset.map(
        lambda x: {
            "prompt": [
                {'role': 'system', 'content': "你是一位经验丰富的医生，专门根据住院患者的病历资料进行初步疾病诊断。\n"
                                            "当提供患者的完整病历（包括：主诉、现病史、婚育史、个人史、体格检查、专科检查等）时，请根据以下格式给出初步可能的疾病诊断：\n"
                                            "对于每个可能的疾病，请按以下格式输出：\n"
                                            "## **疾病名称**：\n"
                                            "- **信息来源**（如主诉、体格检查等）：诊断推理过程\n"
                                            "- **信息来源2**：诊断推理过程\n"
                                            "...\n"
                                            "请从病历资料中提取出关键的诊断线索，标明具体的信息来源，并结合推理说明为什么支持该诊断。\n"
                                            "你的目标是提供清晰、有依据的初步诊断，以指导后续的进一步检查和治疗。\n"
                                            },
                {'role': 'user', 'content': f"这里是患者的病历资料：{x['instruction']}\n"
                                            "请根据提供的信息，按照上述指定格式，给出初步可能的疾病诊断。"},
            ],
            "answer": {'role': 'assistant', 'content': x['output']},
        }
    )
    return processed_dataset

def main():
    dataroot = '/data/csyData/dataset'
    model_type = 'Qwen2.5-7B-Instruct'
    model_path = f"/data2/zzyData2/Qwen/{model_type}"
    data_path = f'{dataroot}/medical_mokeqa.json'
    run_name = f"{model_type}-grpo-withoutsft"
    lora_model_path = f"./save/{run_name}/medical_sft_lora"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if not exists(lora_model_path):
        loraconfig = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, loraconfig)
    else:
        model = PeftModel.from_pretrained(
            model, 
            lora_model_path,
            is_trainable=True,
            config=PeftConfig.from_pretrained(lora_model_path)
        )

    dataset = process_data(f'{dataroot}/medical_clean1.json')

    model_dir = "/data/csyData/models/iic/nlp_corom_sentence-embedding_chinese-base-medical"
    emb_model = Model.from_pretrained(model_dir, trust_remote_code=True)
    emb_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    reward_fn = get_reward_fn(model=emb_model, tokenizer=emb_tokenizer)

    trainer = GRPOTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=GRPOConfig(
            output_dir="outputs",
            per_device_train_batch_size=4,
            num_generations=4,
            gradient_accumulation_steps=8,
            max_completion_length=512,
            optim="adamw_torch",
            learning_rate=2e-4,
            save_steps=50,
            lr_scheduler_type="cosine",
            num_train_epochs=3,
            # logging_steps=1,
            bf16=True,
            report_to="wandb",
            run_name=run_name,
            use_vllm=False
        ),
        reward_funcs=[reward_fn],
    )

    trainer.train()
    trainer.model.save_pretrained(lora_model_path)

if __name__ == "__main__":
    main()
