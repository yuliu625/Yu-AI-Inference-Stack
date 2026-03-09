"""
Sources:
    https://github.com/yuliu625/Yu-AI-Inference-Stack/inference_engines/litserve/litserve_launcher.py

References:
    https://lightning.ai/docs/litserve/home

Synopsis:
    基于 LitServe 的启动器。

Notes:
    使用 LitServe 即 Lightning 体系的工具封装模型，快捷实现推理服务化。
"""

from __future__ import annotations
from loguru import logger

import litserve as ls

from typing import TYPE_CHECKING
# if TYPE_CHECKING:





