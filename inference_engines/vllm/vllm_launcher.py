"""
Sources:
    https://github.com/yuliu625/Yu-AI-Inference-Stack/inference_engines/vllm/vllm_launcher.py

References:
    - https://docs.vllm.ai/en/stable/cli/
    - https://docs.vllm.ai/en/stable/configuration/serve_args/

Synopsis:
    VLLM 启动器。

Notes:
    改进的 vllm 启动器，基于配置文件。

    在新版本的 vllm ，目前已经支持通过配置文件启动服务。
    原始方法因独特需求，总是需要频繁修改。当前 launcher 通过解耦配置文件，启动逻辑不再进行修改。

    配置文件仅是 EngineArgs 的转义，会根据需求频繁更新。

    注意:
        - CLI偏好: vllm依然偏好通过 CLI ，配置文件方法为后续提供。
        - 覆盖优先级: vllm 的优先级是: command line > config file values > defaults 。
"""

from __future__ import annotations
from loguru import logger

import subprocess
import os

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from subprocess import CompletedProcess


class VLLMLauncher:
    # ==== 暴露方法。 ====
    @staticmethod
    def start_vllm(
        mode: Literal['chat', 'complete', 'serve', 'bench', 'collect-env', 'run-batch'],
        config_file_path: str,
        additional_args: list[str],
        # HACK: environment variables
        cuda_visible_devices: str,
    ) -> CompletedProcess[str]:
        """
        阻塞启动vllm的方法。

        Args:
            mode (Literal['chat', 'complete', 'serve', 'bench', 'collect-env', 'run-batch']): 启动 vllm 的方式，最常使用的是 serve 。
            config_file_path (str): 配置文件路径。需要符合 vllm config schema 的 YAML 配置文件。
            additional_args (list[str]): 额外指定的参数。
            cuda_visible_devices (str): 常用环境变量设置。运行的显卡编号。

        Returns:
            CompletedProcess[str]: 运行vllm服务。这个实现是阻塞的。
        """
        command = [
            'vllm',  # V1版本中，不再用"python -m vllm.entrypoints.api_server"启动。
        ]
        # set the mode
        command.extend([mode])
        # cli configs
        command.extend(additional_args)  # 约定，一般情况下不再这里进行设置。
        # use config file
        command.extend(['--config', config_file_path])
        # set environment variables
        env = VLLMLauncher.set_vllm_environment_variables(
            cuda_visible_devices=cuda_visible_devices,
        )
        # launch vllm server
        result = subprocess.run(
            args=command,
            text=True,
            env=env,
        )
        return result

    # ==== 工具方法。 ====
    @staticmethod
    def set_vllm_environment_variables(
        cuda_visible_devices: str,
    ) -> dict[str, str]:
        """
        设置只能通过环境变量额外指定而不能通过 vllm 参数传递设置的环境变量。

        Args:
            cuda_visible_devices (str): 运行设备指定。以 str 的编号进行，如 '0,1,2,3' 。
                注意:
                    - 设置需要与 tensor_parallel_size 匹配。
                    - 设置需要被模型支持。

        Returns:
            dict[str, str]: 继承自当前进程并被设置过的环境变量集。
        """
        # 继承当前程序的环境变量。
        env = os.environ.copy()
        # HACK: 未来添加更多可设置的内容。
        env['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
        return env


if __name__ == '__main__':
    # 一次设置以下参数。
    ## 如果有多个模型需要部署，简单的，复制、配置和多次运行该脚本。
    VLLMLauncher.start_vllm(
        mode='serve',
        config_file_path='./config.yaml',
        additional_args=[],
        cuda_visible_devices='0,1,2,3',
    )

