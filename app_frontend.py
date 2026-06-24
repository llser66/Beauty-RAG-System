"""
💄 Beauty Agent — 美妆护肤 AI 助手
RAG 知识库 + Agent 自主推理 + 多模态图文检索
"""

import os
import time
import json
from pathlib import Path
from typing import Optional

import streamlit as st
import requests

# ── 页面配置 ──────────────────────────────────────
st.set_page_config(
    page_title="Beauty Agent · 美妆 AI 助手",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 常量 ──────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8877")
API_TOKEN: Optional[str] = None

# ── 配色（白底专业风格）──────────────────────────
C = {
    "primary": "#2563EB",       # 主色：蓝（仅用于链接/按钮）
    "primary_light": "#3B82F6",
    "primary_dark": "#1D4ED8",
    "accent": "#0891B2",        # 强调色
    "accent_light": "#06B6D4",
    "bg": "#FFFFFF",            # 主背景：纯白
    "bg_card": "#FFFFFF",       # 卡片背景
    "bg_card_hover": "#F8FAFC",
    "surface": "#F8F9FA",       # 浅灰表面
    "text": "#1E293B",          # 正文：深灰
    "text_sec": "#64748B",      # 次要文字
    "text_muted": "#94A3B8",    # 弱化文字
    "border": "#E2E8F0",        # 边框：浅灰
    "border_light": "#F1F5F9",
    "success": "#059669",
    "warning": "#D97706",
    "error": "#DC2626",
    "info": "#2563EB",
}

# ── CSS ─────────────────────────────────────────────
CSS = f"""
<style>
    /* ── 全局 ── */
    .stApp {{ background-color: #FFFFFF; color: #1E293B; }}
    .stApp > header {{ background: transparent; }}
    #MainMenu, footer, .stDeployButton {{ display: none; }}

    /* ── 滚动条 ── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: #F8F9FA; }}
    ::-webkit-scrollbar-thumb {{ background: #CBD5E1; border-radius: 3px; }}

    /* ── 侧边栏 ── */
    section[data-testid="stSidebar"] {{
        background: #F8F9FA;
        border-right: 1px solid #E2E8F0;
        padding-top: 0;
    }}
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: #475569;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
    }}

    /* ── Logo ── */
    .sidebar-logo {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 1.2rem 1rem;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 0.5rem;
    }}
    .sidebar-logo .logo-icon {{ font-size: 2rem; }}
    .sidebar-logo .logo-title {{
        font-size: 1.1rem;
        font-weight: 700;
        color: #1E293B;
    }}
    .sidebar-logo .logo-sub {{
        font-size: 0.7rem;
        color: #64748B;
    }}

    /* ── 统计卡片 ── */
    .stat-card {{
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
    }}
    .stat-card .stat-val {{
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E293B;
    }}
    .stat-card .stat-lbl {{
        font-size: 0.75rem;
        color: #64748B;
    }}

    /* ── 聊天气泡 ── */
    [data-testid="stChatMessage"] {{
        margin-bottom: 0.3rem;
        animation: fadeIn 0.35s ease;
    }}
    [data-testid="stChatMessageContent"] {{
        border-radius: 12px !important;
        padding: 12px 18px !important;
        font-size: 0.92rem !important;
        line-height: 1.7 !important;
    }}
    [data-testid="user-chat-avatar"] + [data-testid="stChatMessageContent"] {{
        background: #2563EB !important;
        color: white !important;
    }}
    [data-testid="assistant-chat-avatar"] + [data-testid="stChatMessageContent"] {{
        background: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        color: #1E293B !important;
    }}
    [data-testid="stChatMessageContent"] p {{ margin-bottom: 0.5rem; }}
    [data-testid="stChatMessageContent"] strong {{ color: #2563EB; }}
    [data-testid="stChatMessageContent"] ul, [data-testid="stChatMessageContent"] ol {{ padding-left: 1.2rem; }}

    /* ── 来源标签 ── */
    .src-tag {{
        display: inline-block;
        font-size: 0.7rem;
        background: #F1F5F9;
        color: #64748B;
        border-radius: 4px;
        padding: 2px 10px;
        margin: 4px 4px 0 0;
        border: 1px solid #E2E8F0;
    }}

    /* ── Agent 思考面板 ── */
    .agent-thought {{
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #2563EB;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 8px 0 4px 0;
        font-size: 0.85rem;
        color: #475569;
    }}
    .agent-thought .step-tool {{
        color: #2563EB;
        font-weight: 600;
    }}
    .agent-thought .step-summary {{
        color: #64748B;
        margin-top: 2px;
    }}

    /* ── 图片网格 ── */
    .img-grid {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 10px;
    }}
    .img-card {{
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        overflow: hidden;
        width: 180px;
        transition: box-shadow 0.2s;
    }}
    .img-card:hover {{
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }}
    .img-card img {{
        width: 100%;
        height: 140px;
        object-fit: cover;
        border-bottom: 1px solid #E2E8F0;
    }}
    .img-card .img-info {{
        padding: 6px 10px;
    }}
    .img-card .img-name {{
        font-size: 0.75rem;
        color: #475569;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .img-card .img-score {{
        font-size: 0.65rem;
        color: #059669;
    }}

    /* ── 聊天输入框 ── */
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"],
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    [data-testid="stMainViewContainer"] {{
        background: #FFFFFF !important;
    }}
    [data-testid="stChatInput"] {{
        background: #FFFFFF !important;
        border: none !important;
        padding: 0.2rem 0 !important;
    }}
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div {{
        background: transparent !important;
        border: none !important;
    }}
    [data-testid="stChatInputTextArea"] {{
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        color: #1E293B !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.9rem !important;
        caret-color: #2563EB !important;
        min-height: 1.5rem !important;
    }}
    [data-testid="stChatInputTextArea"]:focus {{
        border-color: #2563EB !important;
        box-shadow: 0 0 0 2px rgba(37,99,235,0.1) !important;
    }}
    [data-testid="stChatInputSubmitButton"] {{
        color: #2563EB !important;
        background: transparent !important;
    }}

    /* ── 按钮 ── */
    .stButton > button {{
        border-radius: 8px !important;
        background: #2563EB !important;
        color: white !important;
        border: none !important;
        padding: 0.4rem 1.2rem !important;
        font-weight: 500 !important;
    }}
    .stButton > button:hover {{
        background: #1D4ED8 !important;
        box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
    }}
    .stButton > button[kind="secondary"] {{
        background: transparent !important;
        border: 1px solid #CBD5E1 !important;
        color: #475569 !important;
    }}

    /* ── Select / Input ── */
    .stSelectbox > div > div {{
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        color: #1E293B !important;
    }}
    .stTextInput > div > div > input {{
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        color: #1E293B !important;
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        border-bottom: 1px solid #E2E8F0;
    }}
    .stTabs [aria-selected="true"] {{
        color: #2563EB !important;
        border-bottom-color: #2563EB !important;
    }}

    /* ── 分割线 ── */
    hr {{ border-color: #E2E8F0; margin: 0.8rem 0; }}

    /* ── 图片上传（小附件按钮） ── */
    [data-testid="stFileUploader"] {{
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        padding: 4px 12px !important;
        min-height: unset !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        background: #F8FAFC !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        cursor: pointer !important;
        width: auto !important;
    }}
    [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: #2563EB !important;
        background: #F1F5F9 !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] {{
        display: none !important;
    }}
    [data-testid="stFileUploaderDropzoneInput"] {{
        display: none !important;
    }}
    [data-testid="stFileUploaderFile"] {{
        padding: 4px 12px !important;
        margin: 0 !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        background: #F0FDF4 !important;
    }}

    /* ── 动画 ── */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
</style>
"""


# ══════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════

def api_get(path: str, show_error: bool = True) -> Optional[dict]:
    global API_TOKEN
    try:
        h = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}
        r = requests.get(f"{API_BASE}{path}", headers=h, timeout=5)
        if r.ok:
            return r.json()
        if show_error:
            st.error(f"❌ 请求失败 [GET {path}]: HTTP {r.status_code}")
        return None
    except requests.exceptions.ConnectionError:
        if show_error:
            st.error(f"🔌 无法连接后端 ({API_BASE})")
        return None
    except Exception as e:
        if show_error:
            st.error(f"❌ 请求异常: {e}")
        return None


def api_post(path: str, data: dict = None, files: dict = None) -> Optional[dict]:
    global API_TOKEN
    try:
        h = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}
        r = requests.post(
            f"{API_BASE}{path}",
            json=data if not files else None,
            files=files,
            headers=h,
            timeout=120 if files else 60,
        )
        if r.ok:
            return r.json()
        st.error(f"❌ 请求失败 [POST {path}]: HTTP {r.status_code}")
        return None
    except Exception as e:
        st.error(f"❌ 请求异常: {e}")
        return None


def stream_rag(question: str):
    """RAG 流式 SSE 请求"""
    global API_TOKEN
    try:
        h = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}
        with requests.post(
            f"{API_BASE}/api/v1/rag/chat/stream",
            json={"question": question},
            headers=h,
            stream=True,
            timeout=60,
        ) as r:
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                if data["type"] == "text":
                    yield data["content"]
                elif data["type"] == "metadata":
                    st.session_state["_rag_meta"] = data
                elif data["type"] == "error":
                    yield f"❌ {data['content']}"
    except Exception as e:
        yield f"❌ 请求失败: {e}"


def stream_agent(question: str):
    """Agent 流式 SSE 请求，yield (type, content) 元组"""
    global API_TOKEN
    try:
        h = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}
        with requests.post(
            f"{API_BASE}/api/v1/agent/chat/stream",
            json={"question": question},
            headers=h,
            stream=True,
            timeout=120,
        ) as r:
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                ev = json.loads(payload)
                yield ev
    except Exception as e:
        yield {"type": "error", "content": f"❌ 请求失败: {e}"}


# ══════════════════════════════════════════════════════
# 渲染函数
# ══════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        # ── Logo ──
        st.markdown(
            f"""<div class="sidebar-logo">
                <span class="logo-icon">💄</span>
                <div>
                    <div class="logo-title">Beauty Agent</div>
                    <div class="logo-sub">美妆护肤 AI 助手</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # ── 模式选择 ──
        st.markdown("### 🤖 对话模式")
        mode = st.selectbox(
            "模式",
            options=["Agent 模式（推荐）", "RAG 模式（简洁）"],
            label_visibility="collapsed",
            key="mode_selector",
        )
        st.caption(
            "Agent 模式会自主搜索知识库并推理；RAG 模式直接检索回答"
            if "Agent" in mode
            else "RAG 模式直接检索知识库并生成回答"
        )

        st.divider()

        # ── 知识库概览 ──
        st.markdown("### 📊 知识库")

        if "kb_stats" not in st.session_state or st.session_state.get("_refresh"):
            st.session_state.kb_stats = api_get("/api/v1/knowledge/stats") or {}
            st.session_state["_refresh"] = False

        if st.button("🔄 刷新", help="刷新知识库统计"):
            st.session_state["_refresh"] = True
            st.rerun()

        stats = st.session_state.kb_stats
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""<div class="stat-card"><div class="stat-val">{stats.get("text_chunks", "?")}</div>
                <div class="stat-lbl">📄 文本</div></div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""<div class="stat-card"><div class="stat-val">{stats.get("images", "?")}</div>
                <div class="stat-lbl">🖼️ 图片</div></div>""",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"""<div class="stat-card"><div class="stat-val">{stats.get("documents", "?")}</div>
            <div class="stat-lbl">📂 文档</div></div>""",
            unsafe_allow_html=True,
        )

        st.divider()

        # ── 知识管理 ──
        st.markdown("### 📂 知识管理")
        uploaded = st.file_uploader(
            "上传文档/图片",
            type=["txt", "pdf", "docx", "xlsx", "jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            with st.spinner("⏳ 正在处理入库..."):
                r = api_post("/api/v1/knowledge/upload", files={"file": (uploaded.name, uploaded.getvalue())})
                if r:
                    st.success(f"✅ {uploaded.name} 已入库")
                    st.session_state["_refresh"] = True
                    st.rerun()

        if st.button("🔁 重建索引", type="secondary"):
            with st.spinner("⏳ 全量重建中..."):
                r = api_post("/api/v1/knowledge/rebuild")
                if r:
                    st.success("✅ 重建完成")
                    st.rerun()

        st.divider()

        # ── Token ──
        st.markdown("### 🔑 连接")
        global API_TOKEN
        tok_info = api_get("/api/v1/auth/token", show_error=False)
        if tok_info:
            tk = tok_info.get("token", "")
            API_TOKEN = tk
            st.code(tk[:12] + "…" if len(tk) > 12 else tk, language="text")
        status_color = "🟢" if stats else "🔴"
        st.markdown(f"{status_color} 后端: {'在线' if stats else '离线'}")
        st.caption(f"API: {API_BASE}")


def render_chat_area():
    """主聊天区域"""
    is_agent = "Agent" in st.session_state.get("mode_selector", "")

    # ── 标题 ──
    mode_label = "🤖 Agent 模式" if is_agent else "💬 RAG 模式"
    mode_desc = "自主检索知识库 → 多步推理 → 综合回答" if is_agent else "直接检索知识库 → 生成回答"
    st.markdown(
        f"""<div style="display:flex;align-items:baseline;gap:12px;padding:0.5rem 0 1rem 0;
            border-bottom:1px solid {C['border']};margin-bottom:1rem;">
            <div style="font-size:1.3rem;font-weight:700;color:{C['text']};">💬 对话</div>
            <div style="font-size:0.8rem;color:{C['text_sec']};background:{C['surface']};
                padding:2px 12px;border-radius:12px;border:1px solid {C['border']};">{mode_label}</div>
            <div style="font-size:0.8rem;color:{C['text_muted']};">{mode_desc}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── 初始化 ──
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "_agent_steps" not in st.session_state:
        st.session_state["_agent_steps"] = {}

    # ── 历史消息 ──
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                f"""<div style="text-align:center;padding:5rem 0;">
                    <div style="font-size:3rem;margin-bottom:1rem;">💄</div>
                    <div style="font-size:1.2rem;font-weight:600;color:{C['text']};">欢迎使用 Beauty Agent</div>
                    <div style="font-size:0.85rem;color:{C['text_sec']};margin-top:0.5rem;">
                        我是你的美妆护肤 AI 助手<br>可以帮你解答成分、推荐产品、分析肤质、写种草文案
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

                    # Agent 思考步骤
                    if msg["role"] == "assistant" and msg.get("steps"):
                        _render_agent_steps(msg["steps"])

                    # 来源标签
                    sources = msg.get("extra", {}).get("context_sources", [])
                    if sources:
                        tags = " ".join(
                            f'<span class="src-tag">📄 {Path(s).name}</span>'
                            for s in sources[:4]
                        )
                        st.markdown(tags, unsafe_allow_html=True)

                    # 图片
                    imgs = msg.get("extra", {}).get("images", [])
                    if imgs:
                        _render_images(imgs)

    # ── 输入（回车发送 + 图片上传） ──
    st.markdown(f"<hr style='margin:1rem 0;border-color:{C['border']};'>", unsafe_allow_html=True)

    # 图片上传状态
    if "chat_image" not in st.session_state:
        st.session_state.chat_image = None

    # 附件栏（上传按钮 + 已上传图片预览）
    col_upload, col_preview = st.columns([1, 5])
    with col_upload:
        uploaded = st.file_uploader(
            "📷 上传图片", type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed", key="img_uploader",
        )
        if uploaded and uploaded != st.session_state.get("_last_uploaded"):
            st.session_state["_last_uploaded"] = uploaded
            with st.spinner(""):
                r = api_post("/api/v1/chat/upload_image",
                    files={"file": (uploaded.name, uploaded.getvalue())})
                if r:
                    st.session_state.chat_image = r
                    st.rerun()

    with col_preview:
        if st.session_state.chat_image:
            ci = st.session_state.chat_image
            img_url = f"{API_BASE}{ci['url']}"
            c1, c2 = st.columns([1, 8])
            with c1:
                st.image(img_url, width=48)
            with c2:
                st.caption(f"{ci['filename'][:20]}")
                if st.button("✕ 移除", key="rm_img", type="secondary"):
                    st.session_state.chat_image = None
                    st.rerun()

    # 输入框
    query = st.chat_input(
        placeholder="问一个美妆问题，或上传图片让 Agent 分析你的皮肤",
        key="chat_input",
    )

    # 清空按钮
    if st.session_state.messages:
        _, col_clear = st.columns([6, 1])
        with col_clear:
            if st.button("🗑️ 清空对话", type="secondary"):
                st.session_state.messages = []
                st.session_state["_agent_steps"] = {}
                st.session_state.chat_image = None
                st.rerun()

    # ── 处理发送 ──
    if query:
        # 如果有上传图片，把它加入消息
        user_content = query
        current_img = st.session_state.chat_image
        if current_img:
            user_content += f"\n\n[图片已上传: {current_img['path']}]"
            # 清空已上传图片
            st.session_state.chat_image = None

        st.session_state.messages.append({"role": "user", "content": user_content})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                steps_data = []
                images_data = []
                sources_data = []
                answer_parts = []

                if is_agent:
                    # ── Agent 模式 ──
                    step_placeholder = st.empty()
                    thought_lines = []

                    for ev in stream_agent(query):
                        if ev["type"] == "step":
                            tool_name = ev.get("tool", "")
                            args = ev.get("args", {})
                            label = {"search_knowledge_base": "🔍 搜索知识库", "search_images": "🖼️ 搜索图片", "get_knowledge_base_stats": "📊 查看统计"}.get(tool_name, f"🛠️ {tool_name}")
                            q = args.get("query", "")
                            thought_lines.append(f"• {label} {q}")
                            step_placeholder.markdown(
                                f"""<div class="agent-thought">{'<br>'.join(thought_lines)}</div>""",
                                unsafe_allow_html=True,
                            )
                        elif ev["type"] == "step_result":
                            steps_data.append({
                                "tool": ev.get("tool", ""),
                                "result_summary": ev.get("summary", ""),
                            })
                            thought_lines.append(f"  → {ev.get('summary', '')}")
                            step_placeholder.markdown(
                                f"""<div class="agent-thought">{'<br>'.join(thought_lines)}</div>""",
                                unsafe_allow_html=True,
                            )
                        elif ev["type"] == "images":
                            images_data = ev.get("images", [])
                            if images_data:
                                _render_images(images_data)
                        elif ev["type"] == "text":
                            answer_parts.append(ev.get("content", ""))
                            st.markdown("".join(answer_parts))
                        elif ev["type"] == "done":
                            step_placeholder.empty()

                else:
                    # ── RAG 模式 ──
                    answer = st.write_stream(stream_rag(query))
                    answer_parts.append(answer)

                    meta = st.session_state.pop("_rag_meta", {})
                    images_data = meta.get("images", [])
                    sources_data = meta.get("sources", [])

                # 组装最终消息
                final_answer = "".join(answer_parts)

                # Agent 模式：流结束后从 session_state 获取 extra
                if is_agent:
                    pass  # steps already collected above

                msg_entry = {
                    "role": "assistant",
                    "content": final_answer,
                    "steps": steps_data,
                    "extra": {
                        "images": images_data,
                        "context_sources": sources_data,
                    },
                }
                st.session_state.messages.append(msg_entry)
                st.rerun()


def _render_agent_steps(steps: list):
    """渲染 Agent 思考步骤"""
    if not steps:
        return
    with st.expander("🧠 思考过程", expanded=False):
        for s in steps:
            tool = s.get("tool", "")
            summary = s.get("result_summary", "")
            icon = {"search_knowledge_base": "🔍", "search_images": "🖼️", "get_knowledge_base_stats": "📊"}.get(tool, "🛠️")
            st.markdown(f"**{icon} {tool}**")
            if summary:
                st.markdown(f"<span style='color:{C['text_sec']};font-size:0.85rem;'>{summary}</span>", unsafe_allow_html=True)
            st.markdown("")


def _render_images(images: list):
    """渲染图片卡片网格"""
    if not images:
        return
    st.markdown("<div style='margin-top:10px;font-weight:600;font-size:0.85rem;'>🖼️ 相关图片</div>", unsafe_allow_html=True)
    html = '<div class="img-grid">'
    for img in images[:6]:
        path = img.get("path", "")
        name = img.get("filename", "")
        score = img.get("score", 0)
        html += f"""<div class="img-card">
            <img src="{API_BASE}{path}" alt="{name}" onerror="this.style.display='none'">
            <div class="img-info">
                <div class="img-name">{name}</div>
                <div class="img-score">匹配 {score:.3f}</div>
            </div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    render_sidebar()
    render_chat_area()


if __name__ == "__main__":
    main()
