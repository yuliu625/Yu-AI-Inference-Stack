"""
测试 litellm proxy launcher 的运行。
"""

from __future__ import annotations
import pytest
from loguru import logger

from openai import (
    OpenAI,
)
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
)

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class TestLiteLLMProxyLauncher:
    def test_raw_ollama(
        self,
    ):
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
        self,
    ):
        client = OpenAI(
            base_url='http://127.0.0.1:4000',
            api_key='none',
        )
        response = client.chat.completions.create(
            model="ollama-llm",
            messages=[{"role": "user", "content": "What is your model id?"}]
        )
        logger.info(f"Ollama Response: \n{response}")

    def test_ollama_embedding(
        self,
    ):
        embedding_model = OpenAIEmbeddings(
            model="ollama-embedding",
            base_url="http://127.0.0.1:4000",
            api_key='none',
        )
        embedding = embedding_model.embed_query('haha')
        logger.info(f"Ollama Embedding: \n{embedding}")

    def test_vllm_llm(
        self,
    ):
        ...

    def test_vllm_embedding(
        self,
    ):
        ...

    def test_dashscope_llm(
        self,
    ):
        ...

