"""
Sources:
    https://github.com/yuliu625/Yu-AI-Inference-Stack/model_foundations/torch/to_onnx.py

References:

Synopsis:
    将 torch 模型转换为 ONNX Runtime 。

Notes:
    基于 pytorch 原生方法进行转换。
"""

from __future__ import annotations
from loguru import logger

from typing import TYPE_CHECKING
# if TYPE_CHECKING:

