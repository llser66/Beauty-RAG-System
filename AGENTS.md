# Beauty-Agent 项目说明

- 本项目是 FastAPI + Streamlit + Milvus 的美妆护肤 RAG/Agent 应用。
- 本地完整模式：复制 .env.example 为 .env，启动 Docker Compose 的 Milvus 后运行 start.bat。
- 公网演示模式：Dockerfile 使用 DEMO_MODE=true 和 demo_knowledge_base/，不依赖 Milvus，Render 由 deploy/render-start.sh 同时启动 API 与 Streamlit。
- 不提交 .env、API Key、上传文件、知识库运行数据或 Docker volumes。
- 常用检查：python -m pytest -q -p no:cacheprovider tests、python -m py_compile main.py app_frontend.py agent\\tools.py agent\\orchestrator.py rebuild.py、docker compose config --quiet。
