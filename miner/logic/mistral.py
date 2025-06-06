from datasets import load_dataset,Dataset
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig,SFTTrainer
from unsloth import is_bfloat16_supported
from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
from unsloth.chat_templates import get_chat_template
import json
dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
max_seq_length = 2048
def transform_data(data):
  transformed_data = []
  for item in data:
    topic = item["instruction"]  # instruction
    argument = item["chosen_response"]  # output
    conversations = [
        {"role": "user", "content": topic},
        {"role": "assistant", "content": argument}
    ]

    transformed_data.append({"conversations": conversations})  #
  return transformed_data

def format_example(example):
    return {
        "prompt": example["instruction"],  
        "chosen": example["chosen_response"],  
        "rejected": example.get("rejected_response", "I can't answer that."),  
    }

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="teknium/OpenHermes-2.5-Mistral-7B",
    max_seq_length=max_seq_length,
    trust_remote_code=True, 
    dtype=dtype,
    load_in_8bit=False, 
    load_in_4bit=False,
    attn_implementation="flash_attention_2",
    use_flash_attention_2=True,
    device_map="cuda:0",
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    map_eos_token = True, # Maps <|im_end|> to </s> instead
)
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

def load_and_transform_data(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return transform_data(data)
file_path = "/root/G.O.D-test/core/data/cf97adc43245b1bd_train_data.json"

# 加载和转换数据
transformed_data = load_and_transform_data(file_path)

dataset = Dataset.from_list(transformed_data)
system_prompt = "You are a helpful assistant specialized in Darija translation and answering questions. Be precise and accurate." 


dataset = dataset.map(formatting_prompts_func, num_proc=8,batched = True)  
print(dataset[0]["conversations"])
print(dataset[0]["text"])
dataset_dict = dataset.train_test_split(test_size=0.1, seed=2888)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["test"]
#eval_dataset = dataset.shuffle(seed=2888).select(range(int(0.1 * len(dataset))))
#train_dataset = dataset["train"]
model_path = model.config.name_or_path
print(f"Model cache path: {model_path}")
#tokenizer.pad_token = tokenizer.eos_token
#tokenizer.padding_side = "right"
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    modules_to_save = ["lm_head"],
    use_rslora=True,
    lora_dropout=0,
    bias="none",
    loftq_config = None,
    use_gradient_checkpointing = "unsloth",
)

ds_config = {
    "fp16": {"enabled": False},
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"}
    },
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
}


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    eval_dataset=eval_dataset,
    dataset_num_proc = 8,
    dataset_batch_size=1000, 
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 64,
        gradient_accumulation_steps = 1,
        warmup_steps = 30,
        max_steps = 300,
        learning_rate = 1e-4,
        evaluation_strategy="steps",
        eval_steps=50,
        fp16 = False,
        bf16 = True,
        tf32=True,
        logging_steps = 1,
        optim = "adamw_torch_fused",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        max_grad_norm=1.0,
        seed = 2888,
        output_dir = "outputs",
        dataloader_num_workers=8,
        dataloader_pin_memory=True, 
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
        #deepspeed="/root/G.O.D-test/ds.json",
        report_to = "none", # Use this for WandB etc
    ),
)



print("Starting  training...")
trainer_stats = trainer.train()
output_dir = "/root/outputs"
eval_results = trainer.evaluate()
print(f"最终评估结果: {eval_results}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"模型和 tokenizer 已保存到 {output_dir}")
