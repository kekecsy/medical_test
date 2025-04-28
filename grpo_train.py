# grpo_train.py

import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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
from medical_utils import get_mokeqa, create_formatting_record_func, AdmDataset, process_data
from FlagEmbedding import BGEM3FlagModel
from reward_utils import get_reward_fn

from modelscope.models import Model



def main():
    dataroot = '/data/csyData/dataset'
    model_type = 'Qwen2.5-7B-Instruct'
    model_path = f"/data2/zzyData2/Qwen/{model_type}"
    # data_path = f'{dataroot}/medical_mokeqa.json'
    run_name = f"{model_type}-grpo-withoutsft"
    lora_model_path = f"/data/csyData/medical_code/save/{run_name}/medical_sft_lora"

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
            per_device_train_batch_size=2,
            num_generations=2,
            gradient_accumulation_steps=8,
            max_completion_length=1024,
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
