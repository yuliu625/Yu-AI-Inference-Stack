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


class HFLitAPI(ls.LitAPI):
    def setup(self, device) -> None:
        """这里的定义是为了在 predict 方法中使用。"""

    def decode_request(self, request, **kwargs):
        ...

    def predict(self, x, **kwargs):
        ...

    def encode_response(self, output, **kwargs):
        ...


if __name__ == '__main__':
    lit_api = HFLitAPI()
    server = ls.LitServer(
        lit_api=lit_api,
        accelerator='auto',
    )
    # start server
    server.run(
        host='0.0.0.0',
        port=8000,
    )

