# Beauty-Agent 改造进度

## 当前目标

将 Beauty-Agent 改造成可实际使用的美妆护肤 AI 应用：聊天与 Agent 链路稳定、图片分析可用、上传与鉴权安全、界面更有产品感，并明确隐私和医疗边界。

验收条件：

- RAG 与 Agent 接口能够处理空输入、异常和流式事件；
- 用户上传的皮肤图片能够安全传入 Agent 的视觉分析工具；
- 文件名、路径、文件类型和大小经过校验，敏感 Token 不通过公开接口泄露；
- Streamlit 前端完成视觉和交互改造，并正确消费后端事件；
- 关键纯函数有自动化测试，代码差异和运行检查有记录。

## 已完成

- 完成项目基线检查：FastAPI + Streamlit + Milvus + LangChain Agent；
- 修复输入校验、上传路径安全、Agent 事件解析、RAG/Agent 流式输出和异常信息泄露问题；
- 修复对话图片未传入视觉工具的问题，并阻止模型参数读取任意本地路径；
- 增加图片类型/大小校验、EXIF/GPS 元数据清理、Token 保护的读取接口、主动删除和 24 小时默认过期清理；
- 增加上传前隐私同意、模型外部处理说明、医疗风险提示和严重症状就医提醒；
- 更新 README、项目说明、环境模板和前端交互，公开隐私说明接口为 `/api/v1/privacy`；
- 保留并区分用户已有的两个 PNG 删除状态，未覆盖无关工作区改动。

## 已执行检查

- `python -m pytest -q -p no:cacheprovider tests`：15 passed；
- `python -m py_compile ...`：通过；
- 临时端口真实启动验证：`/health` 返回 200，`/api/v1/privacy` 返回 200，受保护图片读取和 Token 查询在无凭证时返回 401；
- `docker compose config --quiet`：通过；`git diff --check`：通过；
- 用户环境已确认 Docker 中 Milvus、etcd、MinIO 和 PostgreSQL 容器运行，`/health` 返回 `status=ok` 且 `milvus=connected`。

## 当前问题与未验证项

- 外部 SiliconFlow、视觉模型和完整 Streamlit 对话链路仍依赖用户 `.env` 中的 API Key、模型下载和运行环境；
- 本轮未在 Codex 环境内执行带真实模型回答的完整 API 联调，仅完成真实启动、健康/权限路由检查、纯函数测试、语法检查和代码级安全校验；
- 已移除 `docker-compose.yml` 中 Docker Compose 已废弃的 `version` 字段；启动脚本现在会轮询健康检查和隐私说明路由，确认新版后端就绪后才启动前端。

## 下一步

1. 用户启动前端后，使用测试图片验证“同意提示 → 上传 → 对话 → 删除”完整流程；
2. 针对真实模型响应补充一次人工验收，确认医疗提示和图片展示符合预期；
3. 若准备部署到公网，再补充 HTTPS、反向代理限流、审计日志脱敏和正式密钥管理。
