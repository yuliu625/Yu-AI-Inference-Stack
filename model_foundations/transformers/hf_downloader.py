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
import asyncio

from huggingface_hub import hf_hub_download, snapshot_download
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

from typing import TYPE_CHECKING, Sequence, Literal
# if TYPE_CHECKING:


class HFDownloaderInterface(ABC):
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
    def __init__(
        self,
        local_model_dir: str,
        local_dataset_dir: str,
        cache_dir: str | None,
        is_only_torch: bool,
        repo_concurrency: int,
        file_concurrency: int,
    ):
        """
        Args:
            local_model_dir: 本地存储仓库这个文件夹的路径。
            local_dataset_dir: 本地存储仓库这个文件夹的路径。
            is_only_torch: 是否仅下载torch相关的文件。默认为了空间和速度，仅下载torch相关。
            # is_batch_download: 是否使用多线程批量下载。多线程下载现在是默认行为。
        """
        self.local_model_dir = Path(local_model_dir)
        self.local_dataset_dir = Path(local_dataset_dir)
        self.cache_dir = Path(cache_dir)
        self._is_only_torch = is_only_torch
        # self.is_batch_download = is_batch_download
        self._repo_concurrency = repo_concurrency
        self._file_concurrency = file_concurrency

        self.set_environment_variables()

    def download_models(
        self,
        repo_ids: Sequence[str],
    ) -> None:
        with ThreadPoolExecutor(max_workers=self._repo_concurrency) as executor:
            executor.map(self.download_datasets, repo_ids)

    def download_datasets(
        self,
        repo_ids: Sequence[str],
    ) -> None:
        with ThreadPoolExecutor(max_workers=self._repo_concurrency) as executor:
            executor.map(self.download_datasets, repo_ids)

    # def try_download_snapshot(
    #     self,
    #     repo_id: str,
    #     repo_type: Literal['model', 'dataset', 'space',]
    # ) -> None:
    #     if self._is_only_torch:
    #         allow_patterns = [
    #             '*.pt', '*.pth', '*.bin',
    #             '*.json', '*.txt', '*.md',
    #             '*.safetensors',
    #             # '*.tar'
    #         ]  # 我不断在检查和总结的torch相关的文件。

    def try_download_model(
        self,
        repo_id: str,
    ) -> None:
        """
        从huggingface上下载模型。

        Args:
            repo_id: huggingface上仓库的id，一般是 "用户名/仓库名" ，可以自动复制的。
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
                    ],  # 我不断在检查和总结的torch相关的文件。
                    max_workers = self._file_concurrency,
                )
            else:
                # 服务器上可以选择这个，完全避免出错。
                snapshot_download(
                    repo_id=repo_id,
                    repo_type='model',
                    local_dir=self.local_model_dir / repo_id,
                    cache_dir=self.cache_dir,
                    max_workers=self._file_concurrency,
                )
            logger.success(f"Save {repo_id} to {self.local_model_dir / repo_id}")
        except Exception as e:
            logger.exception(f"{e}")

    def try_download_dataset(
        self,
        repo_id: str,
    ) -> None:
        """
        从huggingface上下载数据集。
        相比较下载模型，其实仅多了一个kwarg指定仓库类型为数据集。

        Args:
            repo_id: huggingface上仓库的id，一般是 "用户名/仓库名" ，可以自动复制的。
        """
        try:
            # 下载数据集
            snapshot_download(
                repo_id=repo_id,
                repo_type='dataset',  # 如果是数据集
                local_dir=self.local_dataset_dir / repo_id,
                cache_dir=self.cache_dir,
                max_workers=self._file_concurrency,
            )
            logger.success(f"Save {repo_id} to {self.local_dataset_dir / repo_id}")
        except Exception as e:
            logger.exception(f"{e}")

    def set_environment_variables(
        self,
        hf_endpoint: str | None,
        hf_home: str | None,
    ) -> None:
        if hf_endpoint:
            # using a mirror
            os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
        if hf_home:
            os.environ['HF_HOME'] = "~/.cache/huggingface"


if __name__ == "__main__":
    # downloader = HFDownloader(
    #     local_model_dir=r"D:/model/",
    #     local_dataset_dir=r"D:/dataset/",
    #     is_only_torch=True,
    #     # is_batch_download=True,
    # )

    # 仓库id。
    model_repo_ids = [r"Qwen/Qwen2.5-0.5B-Instruct", r"Qwen/Qwen2-VL-2B-Instruct"]
    dataset_repo_ids = [r"HuggingFaceTB/smoltalk"]
    locaL_space_repo_ids = [r"HuggingFaceTB/smoltalk"]

    # asyncio.run(downloader.download_models(model_repo_ids))
    # asyncio.run(downloader.download_datasets(dataset_repo_ids))

