"""
Sources:
    https://github.com/yuliu625/Yu-AI-Inference-Stack/inference_engines/litserve/litserve_launcher.py

References:
    https://lightning.ai/docs/litserve/home

Synopsis:
    基于 LitServe 的启动器。

Notes:
    使用 LitServe 即 Lightning 体系的工具封装模型，快捷实现推理服务化。

    LitServe 继承了 Lightning 的 Hooks 风格，构建方法依然是填入对应逻辑。
"""

from __future__ import annotations
from loguru import logger

import litserve as ls

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class HFLitAPI(ls.LitAPI):
    """
    以下实现方法等同于 pipeline 的流程。
    """
    def setup(self, device) -> None:
        """
        这里的定义是为了在 predict 方法中使用。
        """

    def decode_request(self, request, **kwargs):
        """
        解码 request 。CPU 与 IO 密集任务。
        """

    def predict(self, x, **kwargs):
        """
        对主要内容进行推理。可进行批量化处理进行加速。
        """

    def encode_response(self, output, **kwargs):
        """
        封装结果，返回响应。
        """


if __name__ == '__main__':
    lit_api = HFLitAPI(
        max_batch_size=1,
        api_path='/predict',
    )
    server = ls.LitServer(
        lit_api=lit_api,
        accelerator='auto',
    )
    # start server
    server.run(
        host='0.0.0.0',
        port=8000,
    )

