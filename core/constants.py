import os

from dotenv import load_dotenv


load_dotenv()

VERSION_KEY = 61_000
# Default NETUID if not set in environment
DEFAULT_NETUID = 56

try:
    NETUID = int(os.getenv("NETUID", DEFAULT_NETUID))
except (TypeError, ValueError):
    NETUID = DEFAULT_NETUID

MINER_DOCKER_IMAGE = "weightswandering/tuning_miner:latest"
MINER_DOCKER_IMAGE_DIFFUSION = "diagonalge/diffusion_miner:latest"
VALIDATOR_DOCKER_IMAGE = "weightswandering/tuning_vali:latest"
VALIDATOR_DOCKER_IMAGE_DIFFUSION = "diagonalge/tuning_validator_diffusion:latest"

CONTAINER_EVAL_RESULTS_PATH = "/aplp/evaluation_results.json"

CONFIG_DIR = "core/config/"
OUTPUT_DIR = "core/outputs/"
CACHE_DIR = "~/.cache/huggingface"
CACHE_DIR_HUB = os.path.expanduser("~/.cache/huggingface/hub")
DIFFUSION_DATASET_DIR = "core/dataset/images"

DIFFUSION_REPEATS = 10
DIFFUSION_DEFAULT_INSTANCE_PROMPT = "lora"
DIFFUSION_DEFAULT_CLASS_PROMPT = "style"

MIN_IMAGE_TEXT_PAIRS = 10
MAX_IMAGE_TEXT_PAIRS = 50

CONFIG_TEMPLATE_PATH_DIFFUSION = CONFIG_DIR + "base_diffusion.toml"
CONTAINER_FLUX_PATH = "/app/flux/unet"
CONFIG_TEMPLATE_PATH = CONFIG_DIR + "base.yml"
CONFIG_TEMPLATE_PATH_3B = CONFIG_DIR + "3b.yml"
CONFIG_TEMPLATE_PATH_7B = CONFIG_DIR + "7b.yml"
CONFIG_TEMPLATE_PATH_14B = CONFIG_DIR + "14b.yml"

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
WANDB_TOKEN = os.getenv("WANDB_TOKEN")

HUGGINGFACE_USERNAME = os.getenv("HUGGINGFACE_USERNAME")
CUSTOM_DATASET_TYPE = "custom"
DPO_DEFAULT_DATASET_TYPE = "chatml.intel"
DPO_DEFAULT_FIELD_PROMPT = "question"
DPO_DEFAULT_FIELD_SYSTEM = "system"
DPO_DEFAULT_FIELD_CHOSEN = "chosen"
DPO_DEFAULT_FIELD_REJECTED = "rejected"
