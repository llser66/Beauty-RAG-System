# 💄 Beauty Agent — 美妆护肤 AI 助手

基于 **RAG + Agent** 架构的美妆护肤智能助手，支持多文档知识库检索、多模态图文搜索、BGE 重排，并提供 Streamlit 交互界面。

## ✨ 功能特性

- **📚 知识库 RAG 检索** — 支持 PDF / Word / Excel / TXT 文档自动解析入库
- **🤖 Agent 自主推理** — ReAct 模式：LLM 自主决定调用工具、观察结果、多轮推理
- **🖼️ 多模态搜索** — CLIP 图文双路检索，用文字找到匹配的产品图片
- **🎯 BGE 重排精排** — 粗召回 + Cross-Encoder 精排，提升检索精度
- **⛓️ 流式输出** — SSE 打字机效果，实时展示 AI 思考过程
- **🔐 Token 鉴权** — Bearer Token 保护 API 端点
- **🐳 Docker 部署** — 一键启动 Milvus 向量数据库

## 🏗️ 架构

```
传统 RAG 模式：
  用户 → 搜索知识库 → 重排 → LLM 生成 → 回答

Agent 模式：
  用户 → LLM 分析意图 → 决定调用工具
       ↘ 搜索知识库 → 观察结果 → 继续推理
       ↘ 搜索图片   → 观察结果 → 继续推理
       ↘ 查看统计   → 观察结果 → 继续推理
       → 综合所有信息 → 生成最终回答
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Docker & Docker Compose（运行 Milvus）

### 1. 克隆项目

```bash
git clone https://github.com/YOUR_USERNAME/Beauty-Agent.git
cd Beauty-Agent
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 [SiliconFlow API Key](https://siliconflow.cn)：

```env
SILICONFLOW_API_KEY=sk-your-api-key-here
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 启动 Milvus 向量数据库

```bash
docker compose up -d
```

### 5. 启动应用

**Windows 用户**（推荐）：
```bash
start.bat
```

**手动启动**：
```bash
# 终端1：启动后端
uvicorn main:app --host 127.0.0.1 --port 8001

# 终端2：启动前端
streamlit run app_frontend.py
```

后端 API: `http://127.0.0.1:8001`
前端界面: `http://localhost:8501`

## 📂 项目结构

```
Beauty-Agent/
├── main.py                  # FastAPI 后端主程序
├── app_frontend.py          # Streamlit 前端界面
├── start.bat                # Windows 一键启动脚本
├── docker-compose.yml       # Milvus 向量数据库
├── requirements.txt         # Python 依赖
├── agent/                   # Agent 核心层
│   ├── orchestrator.py      # ReAct 编排器（思考→行动→观察）
│   └── tools.py             # 工具定义（Function Calling）
├── utils/                   # 工具模块
│   ├── document_loader.py   # 多格式文档解析
│   ├── multimodal.py        # CLIP 多模态向量引擎
│   ├── reranker.py          # BGE-Reranker 精排
│   └── auth.py              # API Token 鉴权
├── knowledge_base/          # 知识库目录（放文档）
├── images/                  # 图片素材目录
└── .env.example             # 环境变量模板
```

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI |
| 前端 | Streamlit |
| LLM | DeepSeek-V3（SiliconFlow） |
| 向量模型 | Qwen3-Embedding-8B |
| 视觉模型 | Qwen3-VL-8B-Instruct |
| 向量数据库 | Milvus |
| RAG 框架 | LangChain |
| 多模态 | CLIP（OpenAI） |
| 重排 | BGE-Reranker-v2-m3 |
| 文档解析 | PyMuPDF / python-docx / openpyxl |

## 📝 License

MIT
