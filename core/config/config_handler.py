import os
import uuid
#import shutil
import re
import toml
import yaml
from fiber.logging_utils import get_logger
from transformers import AutoTokenizer

import core.constants as cst
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import DatasetType
from core.models.utility_models import FileFormat


logger = get_logger(__name__)


def create_dataset_entry(
    dataset: str,
    dataset_type: DatasetType | CustomDatasetType,
    file_format: FileFormat,
) -> dict:
    dataset_entry = {"path": dataset}
    #shutil.copy( f"/root/G.O.D-test/core/data/{os.path.basename(dataset)}", "/workspace/input_data")
    if file_format == FileFormat.JSON:
        dataset_entry = {"path": f"/root/G.O.D-test/core/data/{os.path.basename(dataset)}"}

    if isinstance(dataset_type, DatasetType):
        dataset_entry["type"] = dataset_type.value
    elif isinstance(dataset_type, CustomDatasetType):
        custom_type_dict = {key: value for key, value in dataset_type.model_dump().items() if value is not None}
        dataset_entry.update(_process_custom_dataset_fields(custom_type_dict))
    else:
        raise ValueError("Invalid dataset_type provided.")

    if file_format != FileFormat.HF:
        dataset_entry["ds_type"] = file_format.value
        dataset_entry["data_files"] = [os.path.basename(dataset)]

    return dataset_entry


def update_flash_attention(config: dict, model: str):
    # You might want to make this model-dependent

    if any(keyword in model.lower() for keyword in {"llama-2","llama-3","llama2","llama3", "mistral", "gemma", "pythia", "falcon", "phi", "qwen", "deepseek","neural","vikhr","solar","yi","vicuna"}):
        config["flash_attention"] = True
        config["flash_attention_2"] = True
        config["xformers_attention"] = False

    else:
        config["flash_attention"] = False
        config["flash_attention_2"] = False
        #config["sample_packing"] = False
        #config["eval_sample_packing"] = False
        config["xformers_attention"] = True
    config["modules_to_save"] = ["lm_head"]


    model_name_lower = model.lower() 
    match1 = re.search(r"-(\d+(?:\.\d+)?)([b])", model_name_lower)
    if match1:
        size_str = match1.group(1)
        size = float(size_str)
        if size <= 4:
            config["lora_r"] = 32
            config["lora_alpha"] = 64
            config["micro_batch_size"] = 4
            config["gradient_accumulation_steps"] = 10
        
        if size >= 9:
            config["lora_r"] = 32
            config["lora_alpha"] = 64
            config["micro_batch_size"] = 2
            config["gradient_accumulation_steps"] = 12
        if size >= 20:
            config["micro_batch_size"] = 1
            config["lora_r"] = 64
            config["lora_alpha"] = 128
            config["gradient_accumulation_steps"] = 18    

    return config


def update_model_info(config: dict, model: str, job_id: str = "", expected_repo_name: str | None = None):
    logger.info("WE ARE UPDATING THE MODEL INFO")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        config["special_tokens"] = {"pad_token": tokenizer.eos_token}

    config["base_model"] = model
    config["wandb_runid"] = job_id
    config["wandb_name"] = job_id
    config["hub_model_id"] = f"{cst.HUGGINGFACE_USERNAME}/{expected_repo_name or str(uuid.uuid4())}"

    return config


def save_config(config: dict, config_path: str):
    with open(config_path, "w") as file:
        yaml.dump(config, file)


def save_config_toml(config: dict, config_path: str):
    with open(config_path, "w") as file:
        toml.dump(config, file)


def _process_custom_dataset_fields(custom_type_dict: dict) -> dict:
    if not custom_type_dict.get("field_output"):
        return {
            "type": "completion",
            "field": custom_type_dict.get("field_instruction"),
        }

    processed_dict = custom_type_dict.copy()
    processed_dict.setdefault("no_input_format", "{instruction}")
    if processed_dict.get("field_input"):
        processed_dict.setdefault("format", "{instruction} {input}")
    else:
        processed_dict.setdefault("format", "{instruction}")

    return {"format": "custom", "type": processed_dict}
