"""
Sources:
    https://github.com/yuliu625/Yu-AI-Inference-Stack/model_foundations/torch/to_onnxsim.py

References:

Synopsis:
    使用简化工具进行换为 ONNX Runtime 。

Notes:
    依赖:
    ```bash
    pip install onnx-simplifier
    ```
"""

from __future__ import annotations
from loguru import logger

from typing import TYPE_CHECKING
# if TYPE_CHECKING:
