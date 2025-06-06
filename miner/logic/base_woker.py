from datasets import load_dataset,Dataset
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig,SFTTrainer
from unsloth import is_bfloat16_supported
from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
from unsloth.chat_templates import get_chat_template
import json
import os
dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
max_seq_length = 2048

#os.environ["UNSLOTH_DISABLE_FAST_DOWNLOAD"] = "True"
    #field_input: eng
    #field_instruction: darija
    #field_output: darija_ar
    #format: '{instruction} {input}'
    #no_input_format: '{instruction}'
    #system_format: '{system}'
    #system_prompt: ''

#目前阶段需要解决传参和数据集格式转换问题 根据数据集来定义per_device_train_batch_size和max_steps
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
#不支持模型列表  NousResearch/Yarn-Llama-2-13b-64k
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=max_seq_length,
    trust_remote_code=True, 
    dtype=dtype,
    load_in_8bit=False, 
    load_in_4bit=False,
    #full_finetuning=True,
    #attn_implementation="flash_attention_2",
    #use_flash_attention_2=True,
    device_map="cuda:0",
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "unsloth", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
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
    r=64,
    lora_alpha=128,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    modules_to_save = ["lm_head"],
    use_rslora=True,
    lora_dropout=0,
    bias="none",
    loftq_config = None,
    #finetune_vision_layers     = False, # Turn off for just text!
    #finetune_language_layers   = True,  # Should leave on!
    #finetune_attention_modules = True,  # Attention good for GRPO
    #finetune_mlp_modules       = True,  # SHould leave on always!
    use_gradient_checkpointing = False,
)


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    eval_dataset=eval_dataset,
    dataset_num_proc = 8,
    #dataset_batch_size=500, 
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 72,
        gradient_accumulation_steps = 1,
        #warmup_steps = 100,
        max_steps = 700,
        learning_rate = 2e-4,
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
exit()
