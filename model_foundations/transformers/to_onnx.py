"""
Sources:
    https://github.com/yuliu625/Yu-AI-Inference-Stack/model_foundations/torch/to_onnx.py

References:

Synopsis:
    将 transformer 架构的模型转换为 ONNX Runtime 。

Notes:
    使用 HuggingFace 社区的 Optimum 工具实现转换。

    依赖:
    ```bash
    pip install optimum[exporters]
    ```
"""

from __future__ import annotations
from loguru import logger

from typing import TYPE_CHECKING
# if TYPE_CHECKING:
