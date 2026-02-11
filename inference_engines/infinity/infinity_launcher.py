"""
Sources:
    https://github.com/yuliu625/Yu-AI-Inference-Stack/inference_engines/infinity/infinity_launcher.py

References:
    https://github.com/michaelfeil/infinity

Synopsis:
    infinity 启动器。

Notes:
    使用 infinity 启动 embedding 推理服务。
    由于官方文档缺乏，目前使用

    优势:
        - 前沿支持: 有 clip, clap, colpali 支持，省去统一支持的麻烦。
"""

from __future__ import annotations
from loguru import logger

import subprocess

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from subprocess import CompletedProcess


class InfinityLauncher:
    @staticmethod
    def start_infinity(
        model_id: str,
    ) -> CompletedProcess[str]:
        command = [
            'infinity_emb', 'v2',  # 对于 infinity_emb>=0.034 ，使用 cli v2 。
        ]


if __name__ == '__main__':
    # 一次设置以下参数。
    ## 如果有多个模型需要部署，简单的，复制、配置和多次运行该脚本。
    InfinityLauncher.start_infinity(
        model_id='infinity',
    )

