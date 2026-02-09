"""
Sources:
    https://github.com/yuliu625/Yu-AI-Inference-Stack/inference_engines/sglang/sglang_launcher.py

References:
    - https://docs.sglang.io/advanced_features/server_arguments.html

Synopsis:
    SGLang 启动器。

Notes:
    基于 SGLang 的启动器。

    注意:
        - 配置文件参数区别:
            - 上下文长度: --context-length  --max-model-len
            - 并行度: --tp-size  --tensor-parallel-size
            - 数据并行度: --dp-size  --pipeline-parallel-size
            - 调度策略: --schedule-policy  --scheduler-delay-factor
"""

from __future__ import annotations
from loguru import logger

import subprocess
import os

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from subprocess import CompletedProcess


class SGLangLauncher:
    # ==== 暴露方法。 ====
    @staticmethod
    def start_sglang(
        config_file_path: str,
        additional_args: list[str],
        # HACK: environment variables
        cuda_visible_devices: str,
    ) -> CompletedProcess[str]:
        """
        阻塞启动 SGLang 的方法。

        Args:
            config_file_path (str): 配置文件路径。需要符合 SGLang config schema 的 YAML 配置文件。
            additional_args (list[str]): 额外指定的参数。
            cuda_visible_devices (str): 常用环境变量设置。运行的显卡编号。

        Returns:
            CompletedProcess[str]: 运行 SGLang 服务。这个实现是阻塞的。
        """
        command = [
            'python', '-m', 'sglang.launch_server',
        ]
        # cli configs
        command.extend(additional_args)  # 约定，一般情况下不再这里进行设置。
        # use config file
        command.extend(['--config', config_file_path])
        # set environment variables
        env = SGLangLauncher.set_sglang_environment_variables(
            cuda_visible_devices=cuda_visible_devices,
        )
        # launch sglang server
        result = subprocess.run(
            args=command,
            text=True,
            env=env,
        )
        return result

    # ==== 工具方法。 ====
    @staticmethod
    def set_sglang_environment_variables(
        cuda_visible_devices: str,
    ) -> dict[str, str]:
        """
        设置只能通过环境变量额外指定而不能通过 sglang 参数传递设置的环境变量。

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
    SGLangLauncher.start_sglang(
        config_file_path='./config.yaml',
        additional_args=[],
        cuda_visible_devices='0,1,2,3',
    )

