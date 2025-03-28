import queue
import threading
from uuid import UUID
import torch  # 导入 PyTorch
import time  # 导入 time 模块

from fiber.logging_utils import get_logger
from core.models.utility_models import DiffusionJob
from core.models.utility_models import Job
from core.models.utility_models import JobStatus
from core.models.utility_models import TextJob
from miner.logic.job_handler import start_tuning_local
from miner.logic.job_handler import start_tuning_local_diffusion

logger = get_logger(__name__)


class TrainingWorker:
    def __init__(self, num_gpus=8):
        logger.info("=" * 80)
        logger.info("STARTING A LOCAL TRAINING WORKER")
        logger.info("=" * 80)
        self.job_queue: queue.Queue[Job] = queue.Queue()
        self.job_store: dict[str, Job] = {}
        self.gpu_available: list[bool] = [True] * num_gpus  # 记录每个 GPU 的状态
        self.gpu_lock = threading.Lock()  # 用于保护 gpu_available 列表
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.num_gpus = num_gpus

    def _worker(self):
        while True:
            job = self.job_queue.get()
            if job is None:
                break
            try:
                # 获取空闲 GPU
                gpu_id = self._get_available_gpu()
                if gpu_id is None:
                    logger.warning(f"No available GPUs, waiting for one...")
                    # 等待一段时间，重新尝试获取 GPU
                    time.sleep(5)  # 等待 5 秒
                    self.job_queue.put(job) #重新放回队列
                    continue # 跳出本次循环，等待下次循环
                logger.info(f"Job {job.job_id} assigned to GPU {gpu_id}")

                # 设置 CUDA_VISIBLE_DEVICES 环境变量
                with self.gpu_lock:
                    torch.cuda.set_device(gpu_id)
                    
                #开始训练
                if isinstance(job, TextJob):
                    start_tuning_local(job, gpu_id)
                elif isinstance(job, DiffusionJob):
                    start_tuning_local_diffusion(job, gpu_id)

                job.status = JobStatus.COMPLETED
            except Exception as e:
                logger.error(f"Error processing job {job.job_id}: {str(e)}")
                job.status = JobStatus.FAILED
                job.error_message = str(e)
            finally:
                # 释放 GPU
                if hasattr(locals(), 'gpu_id') and gpu_id is not None:
                    self._release_gpu(gpu_id)
                    logger.info(f"GPU {gpu_id} released after job {job.job_id}")
                self.job_queue.task_done()


    def _get_available_gpu(self) -> int | None:
        """获取可用的 GPU ID，如果没有则返回 None"""
        with self.gpu_lock:
            for i in range(self.num_gpus):
                if self.gpu_available[i]:
                    self.gpu_available[i] = False
                    return i
            return None

    def _release_gpu(self, gpu_id: int):
        """释放 GPU"""
        with self.gpu_lock:
            self.gpu_available[gpu_id] = True

    def enqueue_job(self, job: Job):
        self.job_queue.put(job)
        self.job_store[job.job_id] = job

    def get_status(self, job_id: UUID) -> JobStatus:
        job = self.job_store.get(str(job_id))
        return job.status if job else JobStatus.NOT_FOUND

    def shutdown(self):
        self.job_queue.put(None)
        self.thread.join()
