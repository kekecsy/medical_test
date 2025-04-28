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
from medical_utils import process_data, get_mokeqa, formatting_prompts_func, create_test_func


os.environ['TOKENIZERS_PARALLELISM'] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["MASTER_PORT"] = "29501"

def inference(prompts, model, tokenizer, max_new_tokens=1024):
    """
    执行推理的通用函数
    :param prompts: 要推理的文本列表（支持批量）
    :param max_new_tokens: 最大生成token数
    :return: 生成的文本列表
    """
    # 编码输入
    # inputs = tokenizer(
    #     prompts['instruction'],
    #     return_tensors="pt",
    #     padding=True,
    #     truncation=True,
    #     max_length=512,
    #     add_special_tokens=False
    # ).to(model.device)

    # qaPrompt
    # chat_prompts = [[{'role': 'user', 'content': prompt}] for prompt in prompts]

    # admissionPrompt
    chat_prompts = prompts

    inputs = tokenizer.apply_chat_template(
        chat_prompts,
        tokenize=True,
        return_tensors="pt",
        max_length=2048,
        truncation=True,
        padding=True,
    ).to(model.device)

    generate_kwargs = {
        "input_ids": inputs,
        # "attention_mask": inputs.attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id
    }

    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)

    results = tokenizer.batch_decode(
        outputs[:, inputs.shape[1]:], 
        skip_special_tokens=True
    )
    
    return results

# 使用示例 -------------------------------------------------
if __name__ == "__main__":

    # model_type = 'Qwen2.5-7B-Instruct'
    # model_path = f"/data/csydata/deepseek_test/models/Qwen/{model_type}"
    # # model_path = "/data2/zzyData2/Qwen/Qwen2.5-3B"
    # data_path = '/data/csydata/deepseek_test/datasets/medical/medical_mokeqa.json'
    # lora_model_path = f"./save/{model_type}-dialogtemplate/medical_sft_lora"

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
    
    # dataset = get_mokeqa(data_path,tokenizer)
    dataset = process_data(f'{dataroot}/medical_clean1.json')

    test_prompts = [
        dataset[110]['prompt'],
        dataset[111]['prompt']
    ]
    
    true_result = [
        dataset[110]['answer']['content'],
        dataset[111]['answer']['content']
    ]

    # 没有训练过
    results = inference(test_prompts, model, tokenizer)
    
    # 加载Lora参数
    model = PeftModel.from_pretrained(model, lora_model_path)
    model = model.merge_and_unload()

    # Lora微调过后
    SFTresults = inference(test_prompts, model, tokenizer)

    for prompt, res, sftres, true_res in zip(test_prompts, results, SFTresults, true_result):
        print(f"[Prompt]\n{prompt}")
        print(f"[Response]\n{res}\n{'-'*50}")
        print(f"[SFTResponse]\n{sftres}\n{'-'*50}")
        print(f"[TRUE Response]\n{true_res}\n{'-'*50}")