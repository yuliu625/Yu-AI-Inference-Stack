"""
Sources:

References:

Synopsis:

Notes:
    infinity 的 pip 和 CLI 方法似乎不稳定，这里尝试使用 python API 部署服务。
"""

from __future__ import annotations
from loguru import logger

from infinity_emb import (
    AsyncEngineArray,
    AsyncEmbeddingEngine,
    EngineArgs,
    create_server,
)

from typing import TYPE_CHECKING
# if TYPE_CHECKING:




