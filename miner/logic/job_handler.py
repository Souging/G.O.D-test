import os
import shutil
import uuid
import subprocess
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

import toml
import yaml
from fiber.logging_utils import get_logger
from huggingface_hub import HfApi

from core import constants as cst
from core.config.config_handler import create_dataset_entry
from core.config.config_handler import save_config
from core.config.config_handler import save_config_toml
from core.config.config_handler import update_flash_attention
from core.config.config_handler import update_model_info
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import DatasetType
from core.models.utility_models import DiffusionJob
from core.models.utility_models import FileFormat
from core.models.utility_models import TextJob

HF_CACHE_DIR = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")) 
MAX_CACHE_SIZE_GB = 400
DRY_RUN = False

logger = get_logger(__name__)
def get_dir_size(path: Path) -> int:
    
    total = 0
    for entry in path.glob("**/*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total

def get_oldest_model_dirs(cache_dir: Path) -> List[Tuple[Path, float]]:
   
    model_dirs = []
    for model_dir in cache_dir.glob("*/models--*"):
        if model_dir.is_dir():
            last_access = os.path.getatime(model_dir)
            model_dirs.append((model_dir, last_access))
    
    model_dirs.sort(key=lambda x: x[1])
    return model_dirs

def clean_hf_cache(cache_dir: Path, max_size_gb: int) -> None:
    total_size_bytes = get_dir_size(cache_dir)
    max_size_bytes = max_size_gb * 1024 ** 3 
    if total_size_bytes <= max_size_bytes:
        logger.info(f"当前缓存大小 {total_size_bytes / 1024**3:.2f} GB ≤ {max_size_gb} GB，无需清理。")
        return
    logger.info(f"当前缓存大小 {total_size_bytes / 1024**3:.2f} GB > {max_size_gb} GB，开始清理...")
    model_dirs = get_oldest_model_dirs(cache_dir)
    deleted_size = 0
    for model_dir, _ in model_dirs:
        if total_size_bytes - deleted_size <= max_size_bytes:
            break  
        dir_size = get_dir_size(model_dir)
        if DRY_RUN:
            logger.info(f"[DRY RUN] 将删除: {model_dir} ({dir_size / 1024**3:.2f} GB)")
        else:
            logger.info(f"删除: {model_dir} ({dir_size / 1024**3:.2f} GB)")
            shutil.rmtree(model_dir)
        deleted_size += dir_size
    remaining_size = total_size_bytes - deleted_size
    logger.info(f"清理完成，剩余缓存大小: {remaining_size / 1024**3:.2f} GB")

def read_and_check_file(filename="1.txt"):
    try:
        with open(filename, 'r') as f:
            content = f.read().strip()  
            if content == "1":
                return True
            else:
                return False

    except FileNotFoundError:
        with open(filename, 'w') as f:
            f.write("0")
        return False

@dataclass
class LocalEnvironmentDiffusion:
    huggingface_token: str
    wandb_token: str
    job_id: str

    def to_dict(self) -> dict[str, str]:
        return {"HUGGINGFACE_TOKEN": self.huggingface_token, "WANDB_TOKEN": self.wandb_token, "JOB_ID": self.job_id}
    
    def to_env(self):
        for key, value in self.to_dict().items():
            os.environ[key] = value


@dataclass
class LocalEnvironment:
    huggingface_token: str
    wandb_token: str
    job_id: str
    dataset_type: str
    dataset_filename: str

    def to_dict(self) -> dict[str, str]:
        return {
            "HUGGINGFACE_TOKEN": self.huggingface_token,
            "WANDB_TOKEN": self.wandb_token,
            "JOB_ID": self.job_id,
            "DATASET_TYPE": self.dataset_type,
            "DATASET_FILENAME": self.dataset_filename,
        }
    
    def to_env(self):
        for key, value in self.to_dict().items():
            os.environ[key] = value


def _load_and_modify_config(
    dataset: str,
    model: str,
    dataset_type: DatasetType | CustomDatasetType,
    file_format: FileFormat,
    task_id: str,
    expected_repo_name: str | None,
) -> dict:
    """
    Loads the config template and modifies it to create a new job config.
    """
    patch = cst.CONFIG_DIR + f"{model}.yml"
    patch = cst.CONFIG_TEMPLATE_PATH_14B
    if os.path.exists(patch):
        logger.info(f"Loading {model} config template")
        with open(patch, "r") as file:
            config = yaml.safe_load(file)
    else:
        logger.info("Loading 14B config template")
        with open(cst.CONFIG_TEMPLATE_PATH_14B, "r") as file:
            config = yaml.safe_load(file)
    config["datasets"] = []

    dataset_entry = create_dataset_entry(dataset, dataset_type, file_format)
    config["datasets"].append(dataset_entry)

    config = update_flash_attention(config, model)
    config = update_model_info(config, model, task_id, expected_repo_name)
    config["mlflow_experiment_name"] = dataset

    return config


def _load_and_modify_config_diffusion(model: str, task_id: str, expected_repo_name: str | None = None) -> dict:
    """
    Loads the config template and modifies it to create a new job config.
    """
    logger.info("Loading config template")
    logger.info(cst.CONFIG_TEMPLATE_PATH_DIFFUSION)
    with open(cst.CONFIG_TEMPLATE_PATH_DIFFUSION, "r") as file:
        config = toml.load(file)
    config["pretrained_model_name_or_path"] = model
    config["train_data_dir"] = f"/dataset/images/{task_id}/img/"
    config["huggingface_token"] = cst.HUGGINGFACE_TOKEN
    config["huggingface_repo_id"] = f"{cst.HUGGINGFACE_USERNAME}/{expected_repo_name or str(uuid.uuid4())}"
    return config


def create_job_diffusion(
    job_id: str,
    model: str,
    dataset_zip: str,
    expected_repo_name: str | None,
):
    return DiffusionJob(job_id=job_id, model=model, dataset_zip=dataset_zip, expected_repo_name=expected_repo_name)



def create_job_text(
    job_id: str,
    dataset: str,
    model: str,
    dataset_type: DatasetType | CustomDatasetType,
    file_format: FileFormat,
    expected_repo_name: str | None,
):
    return TextJob(
        job_id=job_id,
        dataset=dataset,
        model=model,
        dataset_type=dataset_type,
        file_format=file_format,
        expected_repo_name=expected_repo_name,
    )


def run_command(cmd, env=None):
    """Run a command with the given environment variables and stream output"""
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        shell=True
    )
    
    for line in iter(process.stdout.readline, ""):
        logger.info(line.strip())
    
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def start_tuning_local_diffusion(job: DiffusionJob):
    logger.info("=" * 80)
    logger.info("STARTING LOCAL DIFFUSION TUNING")
    logger.info("=" * 80)

    config_path = os.path.join(cst.CONFIG_DIR, f"{job.job_id}.toml")

    config = _load_and_modify_config_diffusion(job.model, job.job_id, job.expected_repo_name)
    save_config_toml(config, config_path)

    logger.info(config)

    prepare_dataset(
        training_images_zip_path=job.dataset_zip,
        training_images_repeat=cst.DIFFUSION_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=job.job_id,
    )

    local_env = LocalEnvironmentDiffusion(
        huggingface_token=cst.HUGGINGFACE_TOKEN, wandb_token=cst.WANDB_TOKEN, job_id=job.job_id
    )
    
    env = os.environ.copy()
    env.update(local_env.to_dict())
    
    env.update({
        "CONFIG_DIR": "/dataset/configs",
        "OUTPUT_DIR": "/dataset/outputs",
        "DATASET_DIR":"/dataset/images"
    })

    try:
        if cst.HUGGINGFACE_TOKEN:
            run_command("huggingface-cli login --token " + cst.HUGGINGFACE_TOKEN, env)
        
        sd_scripts_path = os.path.expanduser("~/sd-scripts")  # 调整为你的sd-scripts实际路径
        train_cmd = (
            f"accelerate launch --dynamo_backend no --dynamo_mode default "
            f"--mixed_precision bf16 --num_processes 1 --num_machines 1 "
            f"--num_cpu_threads_per_process 2 {sd_scripts_path}/sdxl_train_network.py "
            f"--config_file {config_path}"
        )
        
        logger.info(f"Running command: {train_cmd}")
        run_command(train_cmd, env)

    except Exception as e:
        logger.error(f"Error processing job: {str(e)}")
        raise

    finally:
        train_data_path = f"{cst.DIFFUSION_DATASET_DIR}/{job.job_id}"
        if os.path.exists(train_data_path):
            logger.info(f"Cleaning up temporary data at {train_data_path}")
            shutil.rmtree(train_data_path)

def start_tuning_local(job: TextJob):  #def start_tuning_local(job: TextJob, gpu_id: int):
    logger.info("=" * 80)
    logger.info("STARTING LOCAL TUNING")
    logger.info("=" * 80)
    #with open("1.txt", 'w') as f:
    #    f.write("1")
    config_filename = f"{job.job_id}.yml"
    config_path = os.path.join(cst.CONFIG_DIR, config_filename)

    config = _load_and_modify_config(
        job.dataset,
        job.model,
        job.dataset_type,
        job.file_format,
        job.job_id,
        job.expected_repo_name,
    )
    save_config(config, config_path)

    logger.info(config)

    logger.info(os.path.basename(job.dataset) if job.file_format != FileFormat.HF else "")

    local_env = LocalEnvironment(
        huggingface_token=cst.HUGGINGFACE_TOKEN,
        wandb_token=cst.WANDB_TOKEN,
        job_id=job.job_id,
        dataset_type=job.dataset_type.value if isinstance(job.dataset_type, DatasetType) else cst.CUSTOM_DATASET_TYPE,
        dataset_filename=os.path.basename(job.dataset) if job.file_format != FileFormat.HF else "",
    )
    
    env = os.environ.copy()
    env.update(local_env.to_dict())
    output_dir = "/root/G.O.D-test/miner_id_24/"
    gitignore_path = os.path.join(output_dir, ".gitignore")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(gitignore_path, "w") as f:
        f.write("README.md\n")
    
    env.update({
        "AWS_ENDPOINT_URL": "https://5a301a635a9d0ac3cb7fcc3bf373c3c3.r2.cloudflarestorage.com",
        "AWS_ACCESS_KEY_ID": "d49fdd0cc9750a097b58ba35b2d9fbed",
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_SECRET_ACCESS_KEY": "02e398474b783af6ded4c4638b5388ceb8079c83bb2f8233d5bcef0e60addba6",
        "CONFIG_DIR": cst.CONFIG_DIR,
        "OUTPUT_DIR": cst.OUTPUT_DIR
    })

    try:
        if job.file_format != FileFormat.HF and os.path.exists(job.dataset):
            data_dir = os.path.join(os.path.dirname(os.path.abspath(cst.CONFIG_DIR)), "data")
            os.makedirs(data_dir, exist_ok=True)
            
            dataset_filename = os.path.basename(job.dataset)
            target_path = os.path.join(data_dir, dataset_filename)
            
            if not os.path.exists(target_path):
                shutil.copy(job.dataset, target_path)
                logger.info(f"Copied dataset to {target_path}")
        if cst.HUGGINGFACE_TOKEN:
            run_command("huggingface-cli login --token " + cst.HUGGINGFACE_TOKEN + " --add-to-git-credential", env)
        
        if cst.WANDB_TOKEN:
            run_command("wandb login " + cst.WANDB_TOKEN, env)
        
        run_command(f"accelerate launch -m axolotl.cli.train {config_path}", env)

    except Exception as e:
        logger.error(f"Error processing job: {str(e)}")
        #with open("1.txt", 'w') as f:
        #    f.write("0")
        shutil.rmtree("/root/G.O.D-test/miner_id_24")
        cache_path = Path(HF_CACHE_DIR)
        clean_hf_cache(cache_path, MAX_CACHE_SIZE_GB)
        logger.info(f"Clean")
        

    finally:
        repo = config.get("hub_model_id", None)
        
        if repo:
            hf_api = HfApi(token=cst.HUGGINGFACE_TOKEN)
            hf_api.update_repo_visibility(repo_id=repo, private=False, token=cst.HUGGINGFACE_TOKEN)
            logger.info(f"Successfully made repository {repo} public")
            shutil.rmtree("/root/G.O.D-test/miner_id_24")
            cache_path = Path(HF_CACHE_DIR)
            clean_hf_cache(cache_path, MAX_CACHE_SIZE_GB)
            logger.info(f"Clean")
            #with open("1.txt", 'w') as f:
            #    f.write("0")
        
