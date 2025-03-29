from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig
from unsloth import is_bfloat16_supported
from unsloth import PatchDPOTrainer
from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn

PatchDPOTrainer()

dataset = load_dataset("json", data_files="/root/G.O.D-test/core/data/cf97adc43245b1bd_train_data.json")
system_prompt = "You are a helpful assistant specialized in Darija translation and answering questions. Be precise and accurate." 


def format_example(example):
    return {
        "prompt": example["instruction"],  
        "chosen": example["chosen_response"],  
        "rejected": example.get("rejected_response", "I can't answer that."),  
    }

def format_example1(example, system_prompt=""):
    instruction = example["darija"]
    input_text = example["eng"]
    output_text = example["darija_ar"]
    if input_text:
        prompt = f"{instruction} {input_text}"
    else:
        prompt = instruction
    

    if system_prompt:
        prompt = f"{system_prompt}\n{prompt}"
    return {
        "prompt": prompt,
        "chosen": output_text,
        "rejected": example.get("rejected_response", "I can't answer that."),
    }


dataset = dataset.map(format_example, num_proc=4)  
train_dataset = dataset["train"]
max_seq_length = 2048
dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="HuggingFaceH4/zephyr-7b-beta",
    max_seq_length=max_seq_length,
    #use_flash_attention_2=True,
    trust_remote_code=True, 
    dtype=dtype,
    load_in_8bit=False, 
    load_in_4bit=False,
    #attn_implementation="flash_attention_2",
    device_map="cuda:0",
)
model_path = model.config.name_or_path
print(f"Model cache path: {model_path}")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    modules_to_save = ["lm_head"],
    use_rslora=True,
    lora_dropout=0.1,
    bias="none",
    loftq_config = None,
    random_state = 2888,
    use_gradient_checkpointing = "unsloth",
)
print(model.config._attn_implementation) 
dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args = DPOConfig(
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 6,
        warmup_ratio = 0.1,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = False,
        bf16 = True,
        logging_steps = 1,
        max_steps = 100,
        optim = "adamw_8bit",
        weight_decay = 0.1,
        lr_scheduler_type = "constant_with_warmup",
        seed = 88,
        output_dir = "outputs",
        warmup_steps=20,
        report_to = "none", # Use this for WandB etc
    ),
    beta = 0.1,
    train_dataset = train_dataset,
    tokenizer = tokenizer,
    max_length = 1024,
    max_prompt_length = 512,
)


print("Starting DPO training...")
dpo_trainer.train()
output_dir = "/root/outputs"

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"模型和 tokenizer 已保存到 {output_dir}")
