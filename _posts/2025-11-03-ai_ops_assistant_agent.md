---
layout: post
title:  "一个基于 LangChain 1.0 和 Streamlit 的智能运维助手"
date:   2025-11-03 18:12:00 +0800
tags: [技术]
---
# 使用 langchain 1.0 (Agent, Tools + RAG) + 本地大模型 + 本地嵌入向量开发


一个基于 LangChain 1.0 和 Streamlit 的智能运维助手，使用 Agent 集成了 RAG（检索增强生成）技术和 func tools call 能力。采用 Ollama 本地大模型，无需 API 密钥，支持自然语言交互处理运维任务。
项目地址: [ai_ops_assistant_agent](https://github.com/pluckhuang/ai_ops_assistant_agent)

## ✨ 功能特性

- 🤖 **智能对话界面** - 基于 Streamlit 的友好交互界面
- 🧠 **本地大模型** - 使用 Ollama (llama3.2:3b)，无需 API 调用费用
- 📊 **AWS 监控** - CloudWatch 集成，实时查询 EC2 CPU 使用率
-  **RAG 文档检索** - FAISS 向量数据库 + LCEL 链式调用
- 🔧 **可扩展工具系统** - 基于 LangChain Agent 的工具调用框架
- 🎯 **多模型支持** - 支持 Ollama/OpenAI/AWS Bedrock/DeepSeek

## 🏗️ 系统架构

```
┌─────────────────────────────────────────┐
│         Streamlit 前端 (app.py)         │
│          聊天界面 + 用户交互             │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│      LangChain Agent (agent_runner.py)  │
│      ChatOllama + create_agent          │
│      根据用户问题智能选择工具            │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌───────────────┐    ┌────────────────────┐
│ RAG Tool      │    │ AWS CloudWatch Tool│
│ (qa_chain)    │    │ (get_ec2_cpu_usage)│
└───────┬───────┘    └────────┬───────────┘
        │                     │
        ▼                     ▼
┌───────────────┐    ┌────────────────────┐
│ FAISS Index   │    │   AWS API          │
│ + Embeddings  │    │   boto3 Client     │
└───────────────┘    └────────────────────┘
```

**核心组件说明：**
- **app.py**: Streamlit UI，处理用户输入输出
- **agent_runner.py**: Agent 执行器，协调工具调用
- **tools.py**: 工具定义（AWS 监控、文档检索）
- **rag_chain.py**: RAG 链实现（LCEL 语法）
- **llm_factory.py**: 多模型支持的 LLM 工厂
- **embedding_factory.py**: 嵌入模型工厂（Ollama/OpenAI/本地）
- **vectorstore_manager.py**: FAISS 向量存储管理

## 🚀 快速开始

### 环境要求

- **Python**: 3.11+
- **Ollama**: 本地安装（用于大模型和嵌入）
- **AWS 凭证**: （可选）用于 CloudWatch 监控功能

### 1. 安装 Ollama

```bash
# macOS
brew install ollama

# 启动 Ollama 服务
ollama serve
```

### 2. 下载所需模型

```bash
# 运行自动化脚本
chmod +x setup_ollama.sh
./setup_ollama.sh

# 或手动下载
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 3. 安装 Python 依赖

```bash
# 创建虚拟环境（推荐）
python3.11 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 4. 环境变量配置

创建 `.env` 文件：

```env
# AWS 配置（可选，用于 CloudWatch 功能）
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1

# OpenAI 配置（可选，用于切换模型）
OPENAI_API_KEY=your_openai_api_key

# DeepSeek 配置（可选）
DEEPSEEK_API_KEY=your_deepseek_api_key

# 数据库配置（可选）
DB_URI=mysql+pymysql://user:password@localhost/dbname
```

### 5. 初始化向量数据库

```bash
# 将文档放入 data/ 目录
# 运行向量化脚本（首次运行自动创建）
python vectorstore_manager.py
```

### 6. 运行应用

```bash
streamlit run app.py
```

访问 `http://localhost:8501` 即可使用。

## 📁 项目结构

```
ai_ops_assistant/
├── app.py                    # Streamlit 主应用
├── agent_runner.py           # Agent 执行器（ChatOllama + tools）
├── rag_chain.py             # RAG 链实现（LCEL 语法）
├── tools.py                 # 工具定义（AWS + QA）
├── llm_factory.py           # LLM 工厂（多模型支持）
├── embedding_factory.py     # Embedding 工厂（Ollama/OpenAI/Local）
├── vectorstore_manager.py   # FAISS 向量存储管理
├── config.yaml              # 全局配置文件
├── setup_ollama.sh          # Ollama 模型下载脚本
├── requirements.txt         # Python 依赖
├── pyproject.toml           # 项目配置（Black/isort/flake8）
├── .pre-commit-config.yaml  # 代码质量检查
├── .env                     # 环境变量（不提交到 Git）
├── data/                    # 文档数据目录
│   └── sample.txt
├── faiss_index/             # FAISS 索引存储
│   └── ollama/
│       └── index.faiss
└── db/                      # 数据库相关
    └── init.sql
```

## 🛠️ 主要组件详解

### 1. Agent Runner (`agent_runner.py`)
```python
# 使用 LangChain 的 create_agent 创建 Agent
agent_runner = create_agent(
    ChatOllama(model="llama3.2:3b"),
    tools=[get_ec2_cpu_usage, qa_chain_tool],
    system_prompt="..."
)
```
- **模型**: ChatOllama (llama3.2:3b)
- **工具**: AWS CloudWatch + RAG 文档检索
- **策略**: 根据用户问题自动选择合适工具

### 2. RAG Chain (`rag_chain.py`)
```python
# LCEL 语法构建 RAG 链
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```
- **技术**: LangChain Expression Language (LCEL)
- **向量库**: FAISS
- **嵌入模型**: Ollama (nomic-embed-text)

### 3. AWS CloudWatch Tool (`tools.py`)
```python
@tool(args_schema=AwsCpuCheckInput)
def get_ec2_cpu_usage(instance_id: str, hours: int = 1):
    """查询 EC2 实例的 CPU 使用率"""
    # 使用 boto3 调用 CloudWatch API
```
- **功能**: 查询 EC2 实例 CPU 使用率
- **参数**: instance_id（必需）、hours（默认 1 小时）
- **认证**: 从 `.env` 读取 AWS 凭证

### 4. 多模型支持 (`llm_factory.py`)
```python
def get_llm(provider="ollama", model_name="llama3.2:3b", temperature=0.2):
    if provider == "ollama": return ChatOllama(...)
    elif provider == "openai": return ChatOpenAI(...)
    elif provider == "aws": return ChatBedrock(...)
    # ...
```
支持的模型提供商：
- **Ollama** (默认): 本地免费，无 API 调用
- **OpenAI**: GPT-3.5/4
- **AWS Bedrock**: Claude 等模型
- **DeepSeek**: 国内 API

### 5. Embedding Factory (`embedding_factory.py`)
```python
def get_embeddings(provider="local"):
    if provider == "ollama": return OllamaEmbeddings(...)
    elif provider == "openai": return OpenAIEmbeddings(...)
    else: return LocalEmbeddings(...)  # SentenceTransformer fallback
```

## 💡 使用示例

### 查询 EC2 CPU 使用率
```
用户: 查询实例 i-1234567890abcdef0 最近 2 小时的 CPU 使用率
助手: [调用 get_ec2_cpu_usage 工具]
      实例 i-1234567890abcdef0 在最近 2 小时的平均 CPU 使用率为 45.2%
```

### 文档检索问答
```
用户: 如何配置 AWS Security Group？
助手: [调用 qa_chain_tool，从向量库检索相关文档]
      根据文档，配置 Security Group 的步骤包括...
```

### 自然语言混合查询
```
用户: 我的服务器 CPU 突然升高了，有什么排查建议吗？
助手: [Agent 自动组合多个工具]
      1. [CloudWatch Tool] 当前 CPU 使用率为 85%
      2. [RAG Tool] 根据运维手册，建议检查...
```

## ⚙️ 配置说明

### `config.yaml` 配置项
```yaml
llm_provider: "ollama"          # LLM 提供商
model_name: "llama3"           # 模型名称
temperature: 0.2               # 温度参数

embedding_options: ["local", "openai"]  # 可用的嵌入模型
embedding_model_local: "all-mpnet-base-v2"
embedding_model_openai: "text-embedding-3-small"
```

### 切换模型提供商
修改 `agent_runner.py`:
```python
# 使用 OpenAI
from langchain_openai import ChatOpenAI
chat_model = ChatOpenAI(model="gpt-4", temperature=0)

# 使用 AWS Bedrock
from langchain_aws import ChatBedrock
chat_model = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
```

## 🔧 开发工具

### 代码质量检查
```bash
# 安装 pre-commit hooks
pre-commit install

# 手动运行检查
pre-commit run --all-files
```

配置了以下工具：
- **autoflake**: 移除未使用的导入
- **isort**: 导入排序
- **black**: 代码格式化
- **flake8**: 代码风格检查

### 调试模式
```bash
# 查看 Agent 执行日志
export LANGCHAIN_TRACING_V2=true
streamlit run app.py
```

## 🤝 贡献指南

欢迎提交 Issues 和 Pull Requests！

## 📄 许可证

MIT License

---

**注意**: 首次运行前请确保：
1. ✅ Ollama 服务已启动 (`ollama serve`)
2. ✅ 已下载所需模型 (`./setup_ollama.sh`)
3. ✅ AWS 凭证已配置（如需使用 CloudWatch 功能）
4. ✅ 文档已放入 `data/` 目录并初始化向量库