"""
Sources:
    https://github.com/yuliu625/Yu-AI-Inference-Stack/gateway/litellm/litellm_proxy_launcher.py

References:
    https://docs.litellm.ai/docs/proxy/config_settings

Synopsis:
    litellm proxy 服务启动器。

Notes:
    以 litellm 作为中间路由，对调用进行管理。
"""

from __future__ import annotations
from loguru import logger

import subprocess
import os

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from subprocess import CompletedProcess


class LiteLLMProxyLauncher:
    # ==== 暴露方法。 ====
    @staticmethod
    def start_litellm_proxy(
        config_file_path: str,
        additional_args: list[str],
    ) -> CompletedProcess[str]:
        """
        阻塞启动 litellm proxy 的方法。

        Args:
            config_file_path (str): 配置文件路径。需要符合 vllm config schema 的 YAML 配置文件。
            additional_args (list[str]): 额外指定的参数。

        Returns:
            CompletedProcess[str]: 运行vllm服务。这个实现是阻塞的。
        """
        command = [
            'litellm',
        ]
        # cli configs
        command.extend(additional_args)  # 约定，一般情况下不再这里进行设置。
        # use config file
        command.extend(['--config', config_file_path])
        # launch litellm proxy server
        result = subprocess.run(
            args=command,
            text=True,
        )
        return result


if __name__ == '__main__':
    # 一次设置以下参数。
    LiteLLMProxyLauncher.start_litellm_proxy(
        config_file_path='./config.yaml',
        additional_args=[],
    )

