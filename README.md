# Beauty Agent

一个面向美妆护肤场景的 AI 知识助手：用自己的成分、产品和内容资料构建知识库，再通过 RAG 或 Agent 模式回答问题。

## 已实现能力

- RAG 问答：Milvus 语义检索、BGE 重排、LLM 生成、来源文件展示；
- Agent 问答：自主调用知识库搜索、图片搜索、知识库统计和皮肤照片分析工具；
- 多模态检索：CLIP 文本搜图片；
- 皮肤照片分析：上传图片后由视觉模型做非医疗性的肤质与护理建议；
- 知识库管理：上传 TXT/PDF/DOCX/XLSX，增量入库或全量重建；
- 美妆素材管理：上传 JPG/JPEG/PNG/WEBP/BMP 图片并建立向量索引；
- 流式输出：后端 SSE，前端实时展示回答和 Agent 工具步骤；
- 安全边界：Bearer Token、文件大小/类型/路径校验、聊天图片不暴露服务器绝对路径；
- 隐私与伦理：上传前同意提示、图片 EXIF/GPS 清理、带 Token 的图片访问、主动删除和自动过期；
- 医疗边界：对孕期、儿童、过敏、用药和紧急症状进行保守提示，不做疾病诊断或药物处方。

## 技术栈

- Python 3.10+
- FastAPI + Uvicorn
- Streamlit
- LangChain / LangChain OpenAI
- Milvus / PyMilvus
- DeepSeek-V3、Qwen3-Embedding、Qwen3-VL
- OpenAI CLIP、BGE-Reranker-v2-m3

## 快速启动

### 1. 准备配置

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，至少配置：

```env
SILICONFLOW_API_KEY=你的 API Key
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
```

如果不填写 `API_BEARER_TOKEN`，后端首次启动会自动生成并写入 `.env`。不要把 `.env` 提交到 Git。

### 2. 安装依赖

```powershell
pip install -r requirements.txt
```

开发测试依赖：

```powershell
pip install pytest
```

### 3. 启动 Milvus

```powershell
docker compose up -d
```

### 4. 启动应用

Windows 推荐直接运行：

```powershell
.\start.bat
```

也可以手动启动：

```powershell
# 终端 1
uvicorn main:app --host 0.0.0.0 --port 8877

# 终端 2
streamlit run app_frontend.py
```

- 前端：http://localhost:8501
- 后端：http://127.0.0.1:8877
- 健康检查：http://127.0.0.1:8877/health

首次启动会扫描 `knowledge_base/`，并按需创建文本和图片索引。CLIP、BGE 重排模型第一次使用时可能需要下载。

## API 概览

除 `/health` 和公开的 `/api/v1/privacy` 外，API 默认需要：

```http
Authorization: Bearer <API_BEARER_TOKEN>
```

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/api/v1/rag/chat` | 非流式 RAG 问答 |
| POST | `/api/v1/rag/chat/stream` | RAG SSE 问答 |
| POST | `/api/v1/agent/chat` | 非流式 Agent 问答 |
| POST | `/api/v1/agent/chat/stream` | Agent SSE 问答 |
| POST | `/api/v1/chat/upload_image` | 上传当前对话图片 |
| GET | `/api/v1/chat/uploads/{file_id}` | 鉴权读取当前对话图片 |
| DELETE | `/api/v1/chat/uploads/{file_id}` | 删除当前对话图片 |
| POST | `/api/v1/knowledge/upload` | 上传知识文档或图片 |
| POST | `/api/v1/knowledge/rebuild` | 全量重建索引 |
| GET | `/api/v1/knowledge/stats` | 查看知识库统计 |
| GET | `/api/v1/privacy` | 查看图片保留和模型处理说明 |

聊天请求示例：

```json
{
  "question": "烟酰胺和视黄醇可以一起用吗？",
  "history": [],
  "image_filename": null
}
```

`image_filename` 只能使用 `/api/v1/chat/upload_image` 返回的 `file_id`，后端会再次校验路径。

## 隐私与使用边界

- 上传皮肤照片前，用户需要确认照片属于本人或已经获得照片本人的授权；
- 图片会重新编码并移除 EXIF/GPS 等元数据，默认保留 24 小时，可通过 `CHAT_UPLOAD_RETENTION_HOURS` 调整；
- 聊天图片不再通过公开静态目录访问，读取和删除都需要 Bearer Token；
- 聊天文本和图片可能发送到 `.env` 中配置的外部模型服务，部署前应确认供应商的数据处理政策；
- Beauty Agent 只提供一般护肤信息，不进行疾病诊断，不替代医生。孕期、儿童、过敏、用药或严重症状应先咨询专业人士；
- 图片分析不推断身份、年龄、种族、健康状态或“美丑”，只讨论与护肤有关的可观察表现；
- 产品和成分建议应结合知识库来源、更新时间和用户自身情况，不应替代医生、药师或产品官方说明。

## 项目结构

```text
Beauty-Agent/
├── main.py                  # FastAPI、RAG 流程、上传与索引
├── app_frontend.py          # Streamlit 产品界面
├── agent/
│   ├── orchestrator.py      # Agent 运行、历史消息、SSE 事件
│   └── tools.py             # 知识库、图片、统计、皮肤分析工具
├── utils/
│   ├── document_loader.py   # TXT/PDF/DOCX/XLSX 解析
│   ├── multimodal.py        # CLIP 图文向量
│   ├── reranker.py          # BGE 重排
│   ├── auth.py              # Bearer Token
│   ├── validation.py        # 输入、文件名、路径校验
│   ├── agent_events.py      # Agent 工具结果结构化解析
│   ├── privacy.py           # 图片元数据清理与临时文件过期清理
│   └── safety.py            # 医疗边界与风险提示
├── tests/                   # 回归测试
├── knowledge_base/          # 本地知识文档
├── images/                  # 图片素材
├── chat_uploads/            # 对话临时图片
├── docker-compose.yml       # Milvus 及依赖
├── rebuild.py               # 命令行重建索引
└── STATUS.md                # 跨会话改造进度
```

## 验证

```powershell
pytest -q tests
python -m py_compile main.py app_frontend.py agent\tools.py agent\orchestrator.py rebuild.py
```

外部模型、SiliconFlow 和 Milvus 没有启动时，接口联调无法完成；这不影响纯函数测试和语法检查。
