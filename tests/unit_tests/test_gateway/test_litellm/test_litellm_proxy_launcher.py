"""
测试 litellm proxy launcher 的运行。
"""

from __future__ import annotations
import pytest
from loguru import logger

from openai import (
    OpenAI,
)
# from langchain_openai import (
#     ChatOpenAI,
#     OpenAIEmbeddings,
# )

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


@pytest.fixture(name='openai_client')
def make_openai_client():
    client = OpenAI(
        base_url='http://127.0.0.1:4000',
        api_key='none',
    )
    return client


def test_raw_ollama():
    client = OpenAI(
        base_url='http://127.0.0.1:11434/v1/',
        api_key='none',
    )
    response = client.chat.completions.create(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": "What is your model id?"}]
    )
    logger.info(f"Raw Ollama Response: \n{response}")


def test_ollama_llm(
    openai_client: OpenAI,
):
    response = openai_client.chat.completions.create(
        model="ollama-llm",
        messages=[{"role": "user", "content": "What is your model id?"}]
    )
    logger.info(f"Ollama Response: \n{response}")


def test_ollama_embedding(
    openai_client: OpenAI,
):
    response = openai_client.embeddings.create(
        input='haha',
        model='ollama-embedding',
    )
    logger.info(f"Ollama Embedding Response: \n{response}")


# def test_vllm_llm(
#     openai_client: OpenAI,
# ):
#     ...
#
#
# def test_vllm_embedding(
#     openai_client: OpenAI,
# ):
#     ...
#
#
# def test_dashscope_llm(
#     openai_client: OpenAI,
# ):
#     ...

