# AI Inference Stack
## 📌 Overview
本项目主要整理了从模型训练到服务化部署的常用工具与工程实践。目的是将开发完成的模型快速转化为稳定、可调用的 API 服务，涵盖了实例管理、资源调度及推理加速等方面的技术实现。


## 🛠 Tech Stack
| 模块 | 功能描述      | 常用组件 / 框架 |
|-----------------------|-----------| --- |
| **Agent Ready**       | Agent 增强  | Function Calling, Tool-use |
| **Gateway**           | 统一协议与流量分发 | OpenAI-compatible API, LiteLLM |
| **Inference Engines** | 高性能推理后端实现 | vLLM, SGLang, Infinity, TEI (Text Embeddings Inference) |
| **Quantization**      | 模型量化与压缩方案 | AWQ, AutoGPTQ, BitsAndBytes, GGUF |
| **Model Foundations** | 基础模型加载与定义 | HuggingFace Transformers, Safetensors |


## 📂 核心模块说明
### 模型基础 (Model Foundations)
这是整个推理栈的底层支持，主要负责模型的加载规范与结构定义：
- **格式统一:** 整理了将 PyTorch 权重转化为 `Safetensors` 的脚本，以实现更快的加载速度和更高的加载安全性。
- **架构适配:** 基于 `transformers` 库，管理不同主流模型的配置文件设置。

### 推理引擎 (Inference Engines)
记录了针对不同任务类型的后端选型:
- **Text Generation:** 侧重于 vLLM 和 SGLang 的配置与部署实践。
- **Embedding/Rerank:** 侧重于高性能向量模型服务（如 Infinity, TEI）。

### 网关与标准化 (Gateway)
解决“多后端、多协议”的接入问题:
- **标准化:** 统一不同推理后端至 OpenAI API 标准，确保与现有科研工具链（如 LangChain, LlamaIndex）无缝对接。

### 量化加速 (Quantization)
针对资源受限场景的显存优化方案:
- 收集了针对不同显存环境的量化脚本，主要包括量化权重转换流程。

### 智能体适配 (Agent Ready)
连接模型与应用逻辑的最后一步:
- 针对大模型 Agent 场景，预处理 Prompt 和后处理解析的常用逻辑。


## 🔗 系列工具箱
该仓库是我个人科研工具链的一部分，你可以配合以下仓库使用:
- **[RAG-Toolkit](https://github.com/yuliu625/Yu-RAG-Toolkit)**: 核心检索增强工具集。
- **[Agent-Development-Toolkit](https://github.com/yuliu625/Yu-Agent-Development-Toolkit)**: 专注于智能体(Agents)逻辑构建。
- **[Deep-Learning-Toolkit](https://github.com/yuliu625/Yu-Deep-Learning-Toolkit)**: 深度学习通用任务底座。
- **[Data-Science-Toolkit](https://github.com/yuliu625/Yu-Data-Science-Toolkit)**: 数据科学与预处理工具。

