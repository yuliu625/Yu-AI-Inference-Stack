"""
Sources:
    https://github.com/yuliu625/Yu-Agent-Development-Toolkit/llm_launchers/vllm_launcher.py
    https://github.com/yuliu625/Yu-CLI-Wrapper/llm_launcher/vllm_launcher.py

References:
    https://docs.vllm.ai/en/stable/cli/serve/

Synopsis:
    VLLM的启动器。

Notes:
    VLLM的设计使得在部署推理服务时，使用CLI是更好的选择。然而大量基于 shell 的输入很不方便，因此我进行了封装。
    这个实现是基于 VLLM V1，主要场景是本地推理服务。

    目的:
        - 封装shell输入: 使用subprocess封装用shell启动和配置vllm的输入。
        - 复用: 自动参数处理。
        - 隔离环境: 独立推理环境，减少开发环境依赖管理复杂度。
"""

from __future__ import annotations

import subprocess
import os

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from subprocess import CompletedProcess


class VLLMLauncher:
    @staticmethod
    def start_vllm(
        env: dict[str, str],
        mode: Literal['serve', 'chat'],  # 目前需求是serve和chat，后续添加其他方法。
        model_name_or_path: str,
        served_model_name: str | None = None,
        host: str | None = None,
        port: int | None = None,
        quantization: str | None = None,
        kv_cache_dtype: str | None = None,
        gpu_memory_utilization: float | None = None,
        tensor_parallel_size: int | None = None,
        max_model_len: int | None = None,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
        lora_modules: dict[str, str] | None = None,
    ) -> CompletedProcess[str]:
        """
        阻塞启动vllm的方法。

        Args:
            env (dict[str, str]): 启动子进程是传递的环境变量。
            mode (Literal['serve', 'chat']): 启动 vllm 的方式，根据文档进行的设置。
            model_name_or_path (str): 模型的路径。这个工具类主要可以简化和灵活设置的地方。
            served_model_name (str, optional): API使用的模型名称。
                在科研环境中，统一设置为 'model' ，避免不需要的双重锁定造成的配置繁琐。
            host (str, optional): url的host部分。我默认为local host。
            port (int, optional): url的端口号。这会在同时运行多个模型时很有用。
            quantization (str, optional):
            kv_cache_dtype (str, optional):
            gpu_memory_utilization:
            tensor_parallel_size:
            max_model_len:
            max_num_batched_tokens:
            max_num_seqs:
            lora_modules (dict[str, str], optional): 加载adapter的具体设置。

        Returns:
            None: 运行vllm服务。这个实现是阻塞的。
        """
        command = [
            'vllm',  # V1版本中，不再用"python -m vllm.entrypoints.api_server"启动。
        ]
        # set the mode
        if mode == 'serve':
            command.extend(['serve'])  # 为了通用性设置这个。
        elif mode == 'chat':
            command.extend(['chat'])
        # set the model path, which is necessary
        command.extend([model_name_or_path])  # new version vllm do not use --model option
        # conditional add other args
        if served_model_name:
            command.extend(['--served-model-name', served_model_name])
        if host:
            command.extend(['--host', host])
        if port:
            command.extend(['--port', str(port)])
        if quantization:
            command.extend(['--quantization', quantization])
        if kv_cache_dtype:
            command.extend(['--kv-cache-dtype', kv_cache_dtype])
        if gpu_memory_utilization:
            command.extend(['--gpu-memory-utilization', str(gpu_memory_utilization)])
        if tensor_parallel_size:
            command.extend(['--tensor-parallel-size', str(tensor_parallel_size)])
        if max_model_len:
            command.extend(['--max-model-len', str(max_model_len)])
        if max_num_batched_tokens:
            command.extend(['--max-num-batched-tokens', str(max_num_batched_tokens)])
        if max_num_seqs:
            command.extend(['--max-num-seqs', str(max_num_seqs)])
        if lora_modules:
            command.extend(['--lora-modules', lora_modules])
        # launch vllm server
        try:
            result = subprocess.run(
                args=command,
                text=True,
                env=env,
            )
            return result
        except Exception as e:
            print(e)

    @staticmethod
    def stop_vllm() -> None:
        """
        停止VLLM服务的运行。

        如果以subprocess.Popen非阻塞方式运行，需要实现停止的方法。
        """
        raise NotImplementedError


if __name__ == '__main__':
    # 环境变量设置。
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    # 一次设置以下参数即可。如果有多个模型需要部署，简单的，复制和配置该文件多次。
    VLLMLauncher.start_vllm(
        env=env,
        mode='serve',
        model_name_or_path=r'',
        served_model_name='model',
        host='127.0.0.1',
        port=8000,
        gpu_memory_utilization=0.1,
        tensor_parallel_size=1,
        max_model_len=None,
        max_num_batched_tokens=None,
        max_num_seqs=None,
        lora_modules=None,
    )

