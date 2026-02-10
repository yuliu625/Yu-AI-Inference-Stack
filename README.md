# AI Inference Stack
## 📌 Overview
This project centralizes common tools and engineering practices for transitioning from model training to production-ready service deployment. The goal is to rapidly transform trained models into stable, callable API services, covering instance management, resource orchestration, and inference acceleration.

## 🛠 Tech Stack
| Module | Functionality | Key Components / Frameworks |
| --- | --- | --- |
| **Agent Ready** | Agent enhancement | Function Calling, Tool-use |
| **Gateway** | Unified protocol & traffic distribution | OpenAI-compatible API, LiteLLM |
| **Inference Engines** | High-performance backends | vLLM, SGLang, Infinity, TEI (Text Embeddings Inference) |
| **Quantization** | Model compression & optimization | AWQ, AutoGPTQ, BitsAndBytes, GGUF |
| **Model Foundations** | Base loading & definitions | HuggingFace Transformers, Safetensors |


## 📂 Core Modules
### Model Foundations
The underlying support for the entire inference stack, focusing on loading standards and structural definitions:
- **Format Uniformity:** Scripts for converting PyTorch weights to `Safetensors` for faster loading speeds and enhanced security.
- **Architecture Adaptation:** Managing configuration settings for various mainstream models based on the `transformers` library.

### Inference Engines
Documented backend selections tailored for different task types:
- **Text Generation:** Focuses on configuration and deployment practices for **vLLM** and **SGLang**.
- **Embedding/Rerank:** Focuses on high-performance vector model services (e.g., **Infinity**, **TEI**).

### Gateway & Standardization
Solving the "multi-backend, multi-protocol" integration challenge:
- **Standardization:** Unifying various inference backends into the **OpenAI API standard**, ensuring seamless integration with existing research toolchains like LangChain and LlamaIndex.

### Quantization & Acceleration
Memory optimization solutions for resource-constrained environments:
- A collection of quantization scripts for different VRAM capacities, focusing on weight conversion workflows.

### Agent Ready
The final step connecting models to application logic:
- Common logic for Prompt preprocessing and response parsing specifically designed for LLM Agent scenarios.


## 🔗 Related Toolkits
This repository is part of my personal research ecosystem. You can use it in conjunction with the following:
- **[RAG-Toolkit](https://github.com/yuliu625/Yu-RAG-Toolkit)**: Core retrieval-augmented toolset.
- **[Agent-Development-Toolkit](https://github.com/yuliu625/Yu-Agent-Development-Toolkit)**: Focused on building logic for Intelligent Agents.
- **[Deep-Learning-Toolkit](https://github.com/yuliu625/Yu-Deep-Learning-Toolkit)**: A general-purpose foundation for deep learning tasks.
- **[Data-Science-Toolkit](https://github.com/yuliu625/Yu-Data-Science-Toolkit)**: Tools for data science and preprocessing.

