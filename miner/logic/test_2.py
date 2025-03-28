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


def format_example(example):
    return {
        "prompt": example["instruction"],  
        "chosen": example["chosen_response"],  
        "rejected": example.get("rejected_response", "I can't answer that."),  
    }


dataset = dataset.map(format_example, num_proc=4)  
train_dataset = dataset["train"]
max_seq_length = 2048
dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="HuggingFaceH4/zephyr-7b-beta",
    max_seq_length=max_seq_length,
    trust_remote_code=True, 
    dtype=dtype,
    load_in_8bit=False, 
    load_in_4bit=False,
    device_map="cuda:0",
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    modules_to_save = ["lm_head"],
    use_rslora=True,
    lora_dropout=0.1,
    bias="none",
    loftq_config = None,
    random_state = 2888,
    #use_gradient_checkpointing = "unsloth",
)

dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args = DPOConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = 3,
        learning_rate = 1e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        max_steps = 10,
        optim = "adamw_torch_fused",
        weight_decay = 0.0,
        lr_scheduler_type = "linear",
        seed = 88,
        output_dir = "outputs",
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
