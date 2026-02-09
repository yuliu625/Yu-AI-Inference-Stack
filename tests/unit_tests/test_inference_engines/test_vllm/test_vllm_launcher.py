"""
测试VLLM的启动和运行。

因为约定启动的是API服务，因此基于网络进行测试。
"""

from __future__ import annotations
import pytest

import subprocess

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


@pytest.mark.parametrize(
    "port", [
    ('12345'),
])
def test_vllm_status(
    port: int,
) -> None:
    result = subprocess.run(
        ['url', f"http://localhost:{port}/health"],
    )
    print(result)


@pytest.mark.parametrize(
    "model_name_or_path, port", [
    (r"", 12345),
])
def test_vllm_via_curl(
    model_name_or_path: str,
    port: int,
) -> None:
    result = subprocess.run(
        [
            'curl', f"http://localhost:{port}/v1/chat/completions",
            '-H', 'Content-Type: application/json',
            '-d', '{"model": "{model_name_or_path}", "message": {"role": "user", "content": "Who are you?"}"}'
        ]
    )
    print(result)


@pytest.mark.parametrize(
    "model_name_or_path, port", [
    (r"", 12345),
])
def test_vllm_via_openai_client(
    model_name_or_path: str,
    port: int,
) -> None:
    from openai import OpenAI
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="None")
    completion = client.chat.completions.create(
        model=model_name_or_path,
        messages=[{"role": "user", "content": "Who are you?"}],
    )
    print(completion)
    # print(completion.choices[0].message.content)

