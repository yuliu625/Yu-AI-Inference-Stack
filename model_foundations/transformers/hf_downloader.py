"""
Sources:
    https://github.com/yuliu625/Yu-AI-Inference-Stack/model_foundations/transformers/hf_download.py

References:
    https://huggingface.co/docs/huggingface_hub/guides/download

Synopsis:
    从 huggingface 上下载指定仓库。

Notes:
    从 huggingface 上下载模型和数据集的方法。

    我因为网络原因构建了这个工具。
    单独配置和运行这个文件，将指定仓库下载到本地。

    重构说明:
        - huggingface-cli: CLI 工具实际上基于 huggingface_hub 系列构造，当前基于 python 的实现理论上更快。
        - 同步实现:
            snapshot_download 当前的实现是基于 httpx 的同步 client 实现，因此多线程部分我通过 ThreadPoolExecutor 而非异步批量请求。
            ThreadPoolExecutor 的最大线程设计是 32，主动设置更大实际会被限制。
            如果还要进行重构优化，就只能基于 subprocess 进行更复杂的控制实现。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from loguru import logger

from huggingface_hub import hf_hub_download, snapshot_download
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

from typing import TYPE_CHECKING, Sequence, Literal
# if TYPE_CHECKING:


class HFDownloaderInterface(ABC):
    """
    必要的下载器需要实现的方法。

    repo_type: Literal['model', 'dataset', 'space',]
    这里必要的实现为 model 和 dataset 。
    """
    @abstractmethod
    def download_models(self, *args, **kwargs):
        """
        Batch download models.
        """

    @abstractmethod
    def download_datasets(self, *args, **kwargs):
        """
        Batch download datasets.
        """


class HFDownloader(HFDownloaderInterface):
    """
    通过 huggingface_hub 的 snapshot_download 方法实现对仓库的下载。

    默认使用多线程加速下载。
    """
    def __init__(
        self,
        local_model_dir: str,
        local_dataset_dir: str,
        cache_dir: str | None,
        is_only_torch: bool,
        repo_concurrency: int,
        file_concurrency: int,
        hf_home: str | None,
        hf_endpoint: str | None,
    ):
        """
        Args:
            local_model_dir (str): 本地显式存储模型仓库的文件夹路径。
            local_dataset_dir (str): 本地显式存储数据集仓库的文件夹路径。
            cache_dir (Union[str, None]): 全局自动管理的 cache directory ，非人类可读。可以不进行设置，从而使用默认目录。
            is_only_torch (bool): 是否仅下载 torch 相关的文件。如果为了空间和速度，可仅下载 torch 相关，但没有运行优化。
            repo_concurrency (int): 一级，并行处理的仓库数量。
            file_concurrency (int): 二级，并行处理的文件数量。按照 python 上限可设置为 32 。
            hf_home (Union[str, None]): huggingface 的根目录。
            hf_endpoint (Union[str, None]): huggingface 的 endpoint 。

            # is_batch_download: 是否使用多线程批量下载。多线程下载现在是默认行为。
        """
        # path processing
        self.local_model_dir = Path(local_model_dir)
        self.local_dataset_dir = Path(local_dataset_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        # set downloading configs
        self._is_only_torch = is_only_torch
        self._repo_concurrency = repo_concurrency
        self._file_concurrency = file_concurrency
        # unstable: set env var
        self.set_environment_variables(
            hf_home=hf_home,
            hf_endpoint=hf_endpoint,
        )

    # ==== 暴露方法。 ====
    def download_models(
        self,
        repo_ids: Sequence[str],
    ) -> None:
        # 使用同步多线程下载。
        with ThreadPoolExecutor(max_workers=self._repo_concurrency) as executor:
            executor.map(self.try_download_model_snapshot, repo_ids)

    # ==== 暴露方法。 ====
    def download_datasets(
        self,
        repo_ids: Sequence[str],
    ) -> None:
        # 使用同步多线程下载。
        with ThreadPoolExecutor(max_workers=self._repo_concurrency) as executor:
            executor.map(self.try_download_dataset_snapshot, repo_ids)

    # ==== 重要方法。 ====
    def try_download_model_snapshot(
        self,
        repo_id: str,
    ) -> None:
        """
        尝试从 huggingface 上下载模型仓库。

        States:
            _is_only_torch (bool): 是否仅下载 torch 相关文件。
                实现方法为指定常见 allow_patterns ，其他方法也可通过指定 ignore_patterns 。
            local_model_dir (Path): 本地显式存储模型仓库的文件夹路径。
            cache_dir (Union[Path, None]): 全局自动管理的 cache directory ，非人类可读。可以不进行设置，从而使用默认目录。
            _repo_concurrency (int): 一级，并行处理的仓库数量。
            _file_concurrency (int): 二级，并行处理的文件数量。按照 python 上限可设置为 32 。

        Args:
            repo_id (str): huggingface 上仓库的 id ，格式为 "用户名/仓库名" ，可自动复制。

        Returns:
            None: 尝试执行下载模型。
                报错会以日志进行记录，但不会中断下载任务。
                该方法约定被多线程方法批量运行，各下载任务独立进行，可重复启动和恢复。
        """
        try:
            # 下载模型
            if self._is_only_torch:
                snapshot_download(
                    repo_id=repo_id,
                    repo_type='model',
                    local_dir=self.local_model_dir / repo_id,  # 基础选项。
                    cache_dir=self.cache_dir,
                    allow_patterns=[
                        '*.pt', '*.pth', '*.bin',
                        '*.json', '*.txt', '*.md',
                        '*.safetensors',
                        # '*.tar'
                    ],  # 常见 torch 相关文件 pattern。
                    max_workers=self._file_concurrency,
                )
            else:
                # 服务器上可以选择这个，完全避免出错。
                # 选择该方法可能会同步下载量化和编译后的模型，从而避免本地编译，加速推理服务化。
                snapshot_download(
                    repo_id=repo_id,
                    repo_type='model',
                    local_dir=self.local_model_dir / repo_id,
                    cache_dir=self.cache_dir,
                    max_workers=self._file_concurrency,
                )
            logger.success(f"Model: Save {repo_id} to {self.local_model_dir / repo_id}")
        except Exception as e:
            logger.exception(f"Model: \n{e}")

    # ==== 重要方法。 ====
    def try_download_dataset_snapshot(
        self,
        repo_id: str,
    ) -> None:
        """
        尝试从 huggingface 上下载数据集仓库。

        States:
            local_dataset_dir (Path): 本地显式存储数据集仓库的文件夹路径。
            cache_dir (Union[Path, None]): 全局自动管理的 cache directory ，非人类可读。可以不进行设置，从而使用默认目录。
            _repo_concurrency (int): 一级，并行处理的仓库数量。
            _file_concurrency (int): 二级，并行处理的文件数量。按照 python 上限可设置为 32 。

        Args:
            repo_id (str): huggingface 上仓库的 id ，格式为 "用户名/仓库名" ，可自动复制。

        Returns:
            None: 尝试执行下载数据集。
                报错会以日志进行记录，但不会中断下载任务。
                该方法约定被多线程方法批量运行，各下载任务独立进行，可重复启动和恢复。
        """
        try:
            # 下载数据集。
            snapshot_download(
                repo_id=repo_id,
                repo_type='dataset',  # 如果是数据集
                local_dir=self.local_dataset_dir / repo_id,
                cache_dir=self.cache_dir,
                max_workers=self._file_concurrency,
            )
            logger.success(f"Dataset: Save {repo_id} to {self.local_dataset_dir / repo_id}")
        except Exception as e:
            logger.exception(f"Dataset: \n{e}")

    # ==== unstable ====
    def set_environment_variables(
        self,
        hf_endpoint: str | None,
        hf_home: str | None,
    ) -> None:
        """
        在当前进程局部设置 environment variables 。

        Features:
            - 对应 huggingface 文档设置方法。

        HACK:
            - 可以将这个方法独立出去，但会脱离进程运行状态。

        Args:
            hf_endpoint (Union[str, None]): huggingface 交互 endpoint ，设置这个可修改镜像。
            hf_home (Union[str, None]): 一次性设置的 huggingface root directory 。

        Returns:
            None: 完成设置并记录日志。
        """
        # 判别式加载 environment variables
        ## 如果被设置，更新 env var 。
        ## 如果为None，不覆盖全局设置。
        if hf_endpoint:
            # using a mirror
            os.environ['HF_ENDPOINT'] = hf_endpoint
        if hf_home:
            os.environ['HF_HOME'] = hf_home
        logger.info(f"HF_ENDPOINT: {os.environ['HF_ENDPOINT']}")
        logger.info(f"HF_HOME: {os.environ['HF_HOME']}")


if __name__ == '__main__':
    # 实例化下载器，约定统一配置。
    downloader = HFDownloader(
        local_model_dir=r"D:/model/",
        local_dataset_dir=r"D:/dataset/",
        cache_dir=None,  # 使用全局设置。
        is_only_torch=True,
        repo_concurrency=8,  # 可能会完全阻塞当前机器，根据情况再调整。
        file_concurrency=32,  # 设置为 python 的最大，由 CPU 核心数量自动降低。
        hf_endpoint="https://hf-mirror.com",  # 常用 mirror 之一。
        hf_home=None,  # 使用全局设置。
    )
    # 设置目标仓库 id 。
    model_repo_ids = [r"Qwen/Qwen2.5-0.5B-Instruct", r"Qwen/Qwen2-VL-2B-Instruct"]
    dataset_repo_ids = [r"HuggingFaceTB/smoltalk"]
    # 以运行当前脚本的方法，批量执行下载。
    ## models
    downloader.download_models(
        repo_ids=model_repo_ids,
    )
    ## datasets
    downloader.download_datasets(
        repo_ids=dataset_repo_ids,
    )

