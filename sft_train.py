# train_grpo.py
import json
import os
import torch
from torch.utils.data import DataLoader
from os.path import exists
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import (TrainingArguments, 
                        AutoModelForCausalLM, 
                        AutoTokenizer, 
                        BitsAndBytesConfig, 
                        default_data_collator,
                        )
import sglang as sgl
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import Accelerator
from medical_utils import get_questions2, get_questions2_hard, get_mokeqa, formatting_prompts_func, create_formatting_func


os.environ['TOKENIZERS_PARALLELISM'] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["MASTER_PORT"] = "29501"


# from utils import (correctness_reward_func, 
#                    int_reward_func, 
#                    strict_format_reward_func, 
#                    soft_format_reward_func, 
#                    xmlcount_reward_func)

dataroot = '/data/csyData/dataset'
model_type = 'Qwen2.5-7B-Instruct'
model_path = f"/data2/zzyData2/Qwen/{model_type}"
# model_path = "/data2/zzyData2/Qwen/Qwen2.5-3B"
data_path = f'{dataroot}/medical_mokeqa.json'
run_name = f"{model_type}-sft"
lora_model_path = f"./save/{run_name}/medical_sft_lora"

# quantization_config = BitsAndBytesConfig(
#         bnb_4bit_use_double_quant=True,  # 使用双量化以进一步减少内存
#         bnb_4bit_quant_type="nf4",       # 4-bit 量化类型（nf4 或 fp4）
#         bnb_4bit_compute_dtype=torch.float16  # 计算时使用 float16
#         )

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # 更均衡的分配策略
    # quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)


tokenizer = AutoTokenizer.from_pretrained(model_path)
# EOS_TOKEN = tokenizer.eos_token
# optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)


# TODO 加载 Lora 参数
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

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}")

dataset = get_mokeqa(data_path,tokenizer)

# response_template = " ### Answer:"
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

my_formatting_prompts_func = create_formatting_func(tokenizer)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    formatting_func = my_formatting_prompts_func,
    data_collator=collator,
    args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=1,
        # gradient_accumulation_steps=8,
        optim="adamw_torch",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=10,
        logging_steps=10,
        bf16=True,
        report_to="wandb",
        run_name=f"{model_type}-system-dialogtemplate",
        # gradient_checkpointing=True
    )
)

# 打印一下前5个样本
# for i, batch in enumerate(trainer.get_train_dataloader()):
#     print(f"Batch {i}:")
#     print("Input IDs:", batch['input_ids'])
#     print("Attention Mask:", batch['attention_mask'])
#     print("Labels:", batch['labels'])
#     if i >= 5:  # 打印前 5 个批次
#         break


print(f"Model device: {model.device}")
sample = next(iter(trainer.get_train_dataloader()))
print(f"Data device: {sample['input_ids'].device}")

trainer.train()

trainer.model.save_pretrained(lora_model_path)
