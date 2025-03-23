import os
import shutil
import uuid
import subprocess
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


logger = get_logger(__name__)
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
        return True

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
    logger.info("Loading config template")
    with open(cst.CONFIG_TEMPLATE_PATH, "r") as file:
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

    # 准备数据集
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
    
    # 设置环境变量
    env = os.environ.copy()
    env.update(local_env.to_dict())
    
    # 添加配置目录环境变量
    env.update({
        "CONFIG_DIR": "/dataset/configs",
        "OUTPUT_DIR": "/dataset/outputs",
        "DATASET_DIR":"/dataset/images"
    })

    try:
        # 登录到HuggingFace（如果需要）
        if cst.HUGGINGFACE_TOKEN:
            run_command("huggingface-cli login --token " + cst.HUGGINGFACE_TOKEN, env)
        
        # 构建与Docker中相同的命令
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
        # 清理临时数据（如果需要）
        train_data_path = f"{cst.DIFFUSION_DATASET_DIR}/{job.job_id}"
        if os.path.exists(train_data_path):
            logger.info(f"Cleaning up temporary data at {train_data_path}")
            shutil.rmtree(train_data_path)

def start_tuning_local(job: TextJob):
    logger.info("=" * 80)
    logger.info("STARTING LOCAL TUNING")
    logger.info("=" * 80)
    with open("1.txt", 'w') as f:
        f.write("1")
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
    
    # 设置环境变量
    env = os.environ.copy()
    env.update(local_env.to_dict())
    
    # 添加AWS环境变量
    env.update({
        "AWS_ENDPOINT_URL": "https://5a301a635a9d0ac3cb7fcc3bf373c3c3.r2.cloudflarestorage.com",
        "AWS_ACCESS_KEY_ID": "d49fdd0cc9750a097b58ba35b2d9fbed",
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_SECRET_ACCESS_KEY": "02e398474b783af6ded4c4638b5388ceb8079c83bb2f8233d5bcef0e60addba6",
        "CONFIG_DIR": cst.CONFIG_DIR,
        "OUTPUT_DIR": cst.OUTPUT_DIR
    })

    try:
        # 复制数据集文件（如果不是HF类型）
        if job.file_format != FileFormat.HF and os.path.exists(job.dataset):
            data_dir = os.path.join(os.path.dirname(os.path.abspath(cst.CONFIG_DIR)), "data")
            os.makedirs(data_dir, exist_ok=True)
            
            dataset_filename = os.path.basename(job.dataset)
            target_path = os.path.join(data_dir, dataset_filename)
            
            if not os.path.exists(target_path):
                shutil.copy(job.dataset, target_path)
                logger.info(f"Copied dataset to {target_path}")

        # 登录到HuggingFace
        if cst.HUGGINGFACE_TOKEN:
            run_command("huggingface-cli login --token " + cst.HUGGINGFACE_TOKEN + " --add-to-git-credential", env)
        
        # 登录到W&B
        if cst.WANDB_TOKEN:
            run_command("wandb login " + cst.WANDB_TOKEN, env)
        
        # 运行Axolotl训练命令
        run_command(f"accelerate launch -m axolotl.cli.train {config_path}", env)

    except Exception as e:
        logger.error(f"Error processing job: {str(e)}")
        repo = config.get("hub_model_id", None)
        if repo:
            hf_api = HfApi(token=cst.HUGGINGFACE_TOKEN)
            hf_api.update_repo_visibility(repo_id=repo, private=False, token=cst.HUGGINGFACE_TOKEN)
            logger.info(f"Successfully made repository {repo} public")
            with open("1.txt", 'w') as f:
                f.write("0")
            shutil.rmtree("/root/G.O.D-test/miner_id_24")
            logger.info(f"清理最后文件夹")
        

    finally:
        repo = config.get("hub_model_id", None)
        
        if repo:
            hf_api = HfApi(token=cst.HUGGINGFACE_TOKEN)
            hf_api.update_repo_visibility(repo_id=repo, private=False, token=cst.HUGGINGFACE_TOKEN)
            logger.info(f"Successfully made repository {repo} public")
            shutil.rmtree("/root/G.O.D-test/miner_id_24")
            logger.info(f"清理最后文件夹")
            with open("1.txt", 'w') as f:
                f.write("0")
        
