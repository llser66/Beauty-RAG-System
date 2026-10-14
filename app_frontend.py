"""Beauty Agent Streamlit 前端。

前端只负责展示和交互，模型、向量库与文件处理全部由 FastAPI 后端完成。
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import quote

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(
    page_title="Beauty Agent · 美妆护肤顾问",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8877").rstrip("/")
API_TOKEN = os.getenv("API_BEARER_TOKEN", "").strip()
CHAT_UPLOAD_RETENTION_HOURS = max(1, int(os.getenv("CHAT_UPLOAD_RETENTION_HOURS", "24")))

COLORS = {
    "ink": "#211A25",
    "ink_soft": "#5E5261",
    "cream": "#F8F4F0",
    "paper": "#FFFDFC",
    "rose": "#B96F7C",
    "rose_light": "#F3DDE0",
    "plum": "#6C3D59",
    "gold": "#B58A52",
    "line": "#E8DFDA",
    "muted": "#8F828B",
    "success": "#2D8A68",
    "danger": "#B94A58",
}

CSS = f"""
<style>
    :root {{
        --ink: {COLORS['ink']};
        --ink-soft: {COLORS['ink_soft']};
        --cream: {COLORS['cream']};
        --paper: {COLORS['paper']};
        --rose: {COLORS['rose']};
        --rose-light: {COLORS['rose_light']};
        --plum: {COLORS['plum']};
        --gold: {COLORS['gold']};
        --line: {COLORS['line']};
        --muted: {COLORS['muted']};
    }}

    .stApp {{ background: var(--cream); color: var(--ink); }}
    .stApp > header {{ background: transparent; }}
    #MainMenu, footer, .stDeployButton {{ display: none; }}
    [data-testid="stAppViewContainer"] {{ background: var(--cream); }}
    [data-testid="stHeader"] {{ background: transparent; }}
    ::selection {{ background: var(--rose-light); color: var(--plum); }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #211A25 0%, #2E2230 100%);
        border-right: 0;
    }}
    section[data-testid="stSidebar"] * {{ color: #F9F2F1; }}
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: #D9BBC5 !important;
        font-size: .68rem;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin: 1.15rem 0 .55rem;
    }}
    section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.12); }}
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
        background: rgba(255,255,255,.08) !important;
        border: 1px dashed rgba(255,255,255,.28) !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] * {{ color: #F9F2F1 !important; }}
    section[data-testid="stSidebar"] .stButton > button {{
        background: rgba(255,255,255,.10) !important;
        border: 1px solid rgba(255,255,255,.18) !important;
        color: #FFF8F7 !important;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{ background: rgba(255,255,255,.18) !important; }}
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background: rgba(255,255,255,.1) !important;
        border-color: rgba(255,255,255,.18) !important;
    }}

    .brand {{ display:flex; align-items:center; gap:.75rem; padding:.35rem .1rem 1.1rem; }}
    .brand-mark {{
        width: 42px; height: 42px; display:flex; align-items:center; justify-content:center;
        border-radius: 14px; background: linear-gradient(135deg, #E9B5BF, #A76582);
        color: #FFF; font-size: 1.25rem; box-shadow: 0 8px 22px rgba(0,0,0,.18);
    }}
    .brand-title {{ font-family: Georgia, serif; font-size: 1.2rem; font-weight: 700; color:#FFF8F7; }}
    .brand-sub {{ font-size:.7rem; color:#D9BBC5; margin-top:.1rem; letter-spacing:.04em; }}

    .sidebar-stat {{
        background: rgba(255,255,255,.075); border:1px solid rgba(255,255,255,.12);
        border-radius:14px; padding:.8rem .7rem; min-height:76px;
    }}
    .sidebar-stat-value {{ color:#FFF8F7; font-size:1.35rem; font-weight:700; }}
    .sidebar-stat-label {{ color:#D9BBC5; font-size:.68rem; margin-top:.18rem; }}
    .sidebar-note {{ color:#D9BBC5; font-size:.76rem; line-height:1.65; }}
    .sidebar-status {{ font-size:.75rem; color:#D9BBC5; padding:.6rem .7rem; border-radius:10px; background:rgba(255,255,255,.06); }}

    .topbar {{ display:flex; align-items:center; justify-content:space-between; margin:.25rem 0 1.25rem; }}
    .eyebrow {{ color:var(--rose); font-size:.7rem; letter-spacing:.16em; text-transform:uppercase; font-weight:700; }}
    .page-title {{ color:var(--ink); font-family:Georgia, serif; font-size:2rem; line-height:1.15; margin-top:.22rem; }}
    .page-subtitle {{ color:var(--ink-soft); font-size:.86rem; margin-top:.35rem; }}
    .status-pill {{
        display:inline-flex; align-items:center; gap:.4rem; border:1px solid var(--line);
        background:rgba(255,253,252,.72); border-radius:999px; padding:.45rem .8rem;
        color:var(--ink-soft); font-size:.72rem;
    }}
    .status-dot {{ width:7px; height:7px; border-radius:50%; display:inline-block; background:#C1B7B3; }}
    .status-dot.online {{ background:#41A77D; box-shadow:0 0 0 4px rgba(65,167,125,.12); }}

    .hero {{
        position:relative; overflow:hidden; border-radius:24px; padding:1.6rem 1.8rem;
        background: radial-gradient(circle at 90% 0%, rgba(231,180,190,.72), transparent 35%),
                    linear-gradient(135deg, #FFFDFC 0%, #F4E7E5 100%);
        border:1px solid rgba(185,111,124,.18); box-shadow:0 16px 38px rgba(91,57,71,.08);
        margin-bottom:1rem;
    }}
    .hero::after {{ content:'✦'; position:absolute; right:2.4rem; bottom:-.7rem; color:rgba(185,111,124,.18); font-size:7rem; font-family:Georgia, serif; }}
    .hero-kicker {{ color:var(--rose); font-size:.7rem; letter-spacing:.15em; text-transform:uppercase; font-weight:700; }}
    .hero-title {{ color:var(--ink); font-family:Georgia, serif; font-size:1.7rem; line-height:1.28; max-width:620px; margin:.45rem 0 .5rem; }}
    .hero-copy {{ color:var(--ink-soft); max-width:650px; line-height:1.7; font-size:.86rem; }}
    .feature-row {{ display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1rem; }}
    .feature-chip {{ border:1px solid rgba(185,111,124,.2); background:rgba(255,255,255,.62); padding:.42rem .68rem; border-radius:999px; color:var(--plum); font-size:.72rem; }}

    .section-label {{ color:var(--ink-soft); font-size:.73rem; letter-spacing:.08em; text-transform:uppercase; font-weight:700; margin:.9rem 0 .6rem; }}
    .quick-card {{
        background:rgba(255,253,252,.72); border:1px solid var(--line); border-radius:14px;
        padding:.72rem .8rem; min-height:64px; color:var(--ink-soft); font-size:.77rem; line-height:1.45;
    }}
    .quick-card strong {{ display:block; color:var(--ink); font-size:.82rem; margin-bottom:.2rem; }}

    [data-testid="stChatMessage"] {{ margin-bottom:.55rem; animation:rise .25s ease-out; }}
    [data-testid="stChatMessageContent"] {{
        border-radius:17px !important; padding:.85rem 1rem !important;
        font-size:.91rem !important; line-height:1.72 !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="user-chat-avatar"]) [data-testid="stChatMessageContent"] {{
        background:var(--plum) !important; color:#FFF8F7 !important; border:0 !important;
        box-shadow:0 8px 18px rgba(108,61,89,.14);
    }}
    [data-testid="stChatMessage"]:has([data-testid="assistant-chat-avatar"]) [data-testid="stChatMessageContent"] {{
        background:rgba(255,253,252,.82) !important; color:var(--ink) !important;
        border:1px solid var(--line) !important; box-shadow:0 6px 18px rgba(91,57,71,.05);
    }}
    [data-testid="stChatMessageContent"] strong {{ color:var(--rose) !important; }}
    [data-testid="stChatInput"] {{ background:transparent !important; border:0 !important; }}
    [data-testid="stChatInputTextArea"] {{
        background:var(--paper) !important; color:var(--ink) !important; border:1px solid #DCCFCB !important;
        border-radius:16px !important; padding:.75rem 1rem !important; font-size:.9rem !important;
        box-shadow:0 8px 22px rgba(91,57,71,.06);
    }}
    [data-testid="stChatInputTextArea"]:focus {{ border-color:var(--rose) !important; box-shadow:0 0 0 3px rgba(185,111,124,.12) !important; }}
    .stButton > button {{ border-radius:10px !important; font-weight:600 !important; }}
    .stButton > button[kind="primary"] {{ background:var(--plum) !important; border-color:var(--plum) !important; }}
    .stButton > button[kind="secondary"] {{ background:transparent !important; border:1px solid var(--line) !important; color:var(--ink-soft) !important; }}
    .stTabs [data-baseweb="tab-list"] {{ gap:1.1rem; border-bottom:1px solid var(--line); }}
    .stTabs [aria-selected="true"] {{ color:var(--rose) !important; border-bottom-color:var(--rose) !important; }}
    [data-testid="stFileUploaderDropzone"] {{ background:rgba(255,253,252,.68) !important; border:1px dashed #D9C7C3 !important; border-radius:12px !important; }}
    .source-tag {{ display:inline-block; margin:.25rem .28rem 0 0; padding:.23rem .5rem; border-radius:6px; background:var(--rose-light); color:var(--plum); font-size:.68rem; }}
    .agent-step {{ padding:.62rem .75rem; border-left:3px solid var(--rose); background:rgba(243,221,224,.38); border-radius:0 10px 10px 0; margin:.32rem 0; font-size:.78rem; color:var(--ink-soft); }}
    .agent-step strong {{ color:var(--plum); }}
    .attachment {{ display:flex; align-items:center; gap:.6rem; padding:.45rem .65rem; border:1px solid var(--line); border-radius:10px; background:rgba(255,253,252,.75); margin-bottom:.45rem; }}
    .attachment img {{ width:42px; height:42px; object-fit:cover; border-radius:8px; }}
    .attachment-name {{ color:var(--ink-soft); font-size:.75rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .image-caption {{ color:var(--ink-soft); font-size:.7rem; margin-top:.2rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .empty-state {{ text-align:center; padding:3.2rem 1rem 2.2rem; color:var(--ink-soft); }}
    .empty-icon {{ font-size:2.7rem; color:var(--rose); margin-bottom:.5rem; }}
    .empty-title {{ font-family:Georgia, serif; color:var(--ink); font-size:1.35rem; }}
    .empty-copy {{ font-size:.84rem; line-height:1.7; max-width:530px; margin:.45rem auto 0; }}
    @keyframes rise {{ from {{ opacity:0; transform:translateY(5px); }} to {{ opacity:1; transform:translateY(0); }} }}
</style>
"""


def _init_state() -> None:
    defaults = {
        "messages": [],
        "mode_selector": "Agent 模式",
        "kb_stats": {},
        "chat_image": None,
        "privacy_consent": False,
        "pending_query": None,
        "uploaded_knowledge_marker": None,
        "refresh_stats": True,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _headers() -> dict[str, str]:
    global API_TOKEN
    if not API_TOKEN:
        load_dotenv(override=True)
        API_TOKEN = os.getenv("API_BEARER_TOKEN", "").strip()
    return {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}


def _error_text(response: requests.Response) -> str:
    try:
        body = response.json()
        detail = body.get("detail") if isinstance(body, dict) else None
        if detail:
            return str(detail)
    except (ValueError, requests.RequestException):
        pass
    return f"请求失败（HTTP {response.status_code}）"


def api_get(path: str, show_error: bool = True) -> Optional[dict[str, Any]]:
    try:
        response = requests.get(f"{API_BASE}{path}", headers=_headers(), timeout=8)
        if response.ok:
            data = response.json()
            return data if isinstance(data, dict) else None
        if show_error:
            st.error(_error_text(response))
    except requests.exceptions.ConnectionError:
        if show_error:
            st.error(f"无法连接后端：{API_BASE}")
    except requests.RequestException as exc:
        if show_error:
            st.error(f"网络请求异常：{exc}")
    return None


def api_post(path: str, data: Optional[dict] = None, files: Optional[dict] = None) -> Optional[dict[str, Any]]:
    try:
        response = requests.post(
            f"{API_BASE}{path}",
            json=data if files is None else None,
            files=files,
            headers=_headers(),
            timeout=120 if files else 90,
        )
        if response.ok:
            payload = response.json()
            return payload if isinstance(payload, dict) else None
        st.error(_error_text(response))
    except requests.exceptions.ConnectionError:
        st.error(f"无法连接后端：{API_BASE}")
    except requests.RequestException as exc:
        st.error(f"网络请求异常：{exc}")
    return None


def api_delete(path: str, show_error: bool = True) -> bool:
    try:
        response = requests.delete(f"{API_BASE}{path}", headers=_headers(), timeout=15)
        if response.ok:
            return True
        if show_error:
            st.error(_error_text(response))
    except requests.exceptions.ConnectionError:
        if show_error:
            st.error(f"无法连接后端：{API_BASE}")
    except requests.RequestException as exc:
        if show_error:
            st.error(f"网络请求异常：{exc}")
    return False


def api_get_bytes(path: str, show_error: bool = False) -> Optional[bytes]:
    try:
        response = requests.get(f"{API_BASE}{path}", headers=_headers(), timeout=15)
        if response.ok:
            return response.content
        if show_error:
            st.error(_error_text(response))
    except requests.RequestException as exc:
        if show_error:
            st.error(f"图片读取失败：{exc}")
    return None


def _chat_image_bytes(file_id: Optional[str]) -> Optional[bytes]:
    if not file_id:
        return None
    safe_id = quote(str(file_id), safe="")
    return api_get_bytes(f"/api/v1/chat/uploads/{safe_id}")


def _delete_chat_image(file_id: Optional[str]) -> None:
    if not file_id:
        return
    safe_id = quote(str(file_id), safe="")
    api_delete(f"/api/v1/chat/uploads/{safe_id}", show_error=False)


def _delete_session_images() -> None:
    file_ids = set()
    current = st.session_state.get("chat_image") or {}
    if current.get("file_id"):
        file_ids.add(str(current["file_id"]))
    for message in st.session_state.get("messages", []):
        file_id = (message.get("extra") or {}).get("image_file_id")
        if file_id:
            file_ids.add(str(file_id))
    for file_id in file_ids:
        _delete_chat_image(file_id)


def stream_events(path: str, payload: dict[str, Any], timeout: int = 120) -> Iterator[dict[str, Any]]:
    """读取后端 SSE 事件，并把传输层错误转换成统一事件。"""

    try:
        with requests.post(
            f"{API_BASE}{path}",
            json=payload,
            headers=_headers(),
            stream=True,
            timeout=timeout,
        ) as response:
            if not response.ok:
                yield {"type": "error", "content": _error_text(response)}
                return
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    break
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    yield event
    except requests.exceptions.ConnectionError:
        yield {"type": "error", "content": f"无法连接后端：{API_BASE}"}
    except requests.RequestException as exc:
        yield {"type": "error", "content": f"网络请求异常：{exc}"}


def _render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-mark">✦</div><div><div class="brand-title">Beauty Agent</div><div class="brand-sub">AI BEAUTY COMPANION</div></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("### 对话模式")
        st.selectbox(
            "对话模式",
            ["Agent 模式", "RAG 模式"],
            key="mode_selector",
            label_visibility="collapsed",
        )
        if st.session_state.mode_selector == "Agent 模式":
            st.caption("自主检索、调用工具并综合回答")
        else:
            st.caption("直接检索知识库，回答更简洁")

        st.divider()
        st.markdown("### 知识库概览")
        if st.session_state.refresh_stats or not st.session_state.kb_stats:
            st.session_state.kb_stats = api_get("/api/v1/knowledge/stats", show_error=False) or {}
            st.session_state.refresh_stats = False
        stats = st.session_state.kb_stats

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="sidebar-stat"><div class="sidebar-stat-value">{stats.get("documents", "—")}</div><div class="sidebar-stat-label">源文档</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="sidebar-stat"><div class="sidebar-stat-value">{stats.get("images", "—")}</div><div class="sidebar-stat-label">图片素材</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div class="sidebar-stat" style="margin-top:.5rem"><div class="sidebar-stat-value">{stats.get("text_chunks", "—")}</div><div class="sidebar-stat-label">可检索文本切片</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("刷新知识库状态", use_container_width=True):
            st.session_state.refresh_stats = True
            st.rerun()

        st.divider()
        st.markdown("### 知识管理")
        uploaded = st.file_uploader(
            "上传文档或图片",
            type=["txt", "pdf", "docx", "xlsx", "jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
        )
        if uploaded:
            marker = f"{uploaded.name}:{uploaded.size}"
            if marker != st.session_state.uploaded_knowledge_marker:
                st.session_state.uploaded_knowledge_marker = marker
                with st.spinner("正在解析并入库…"):
                    result = api_post(
                        "/api/v1/knowledge/upload",
                        files={"file": (uploaded.name, uploaded.getvalue())},
                    )
                if result:
                    st.success(f"已入库：{uploaded.name}")
                    st.session_state.refresh_stats = True
                    st.rerun()

        if st.button("重建全部索引", use_container_width=True):
            with st.spinner("正在重建文本和图片索引…"):
                result = api_post("/api/v1/knowledge/rebuild")
            if result:
                st.success("索引重建完成")
                st.session_state.refresh_stats = True
                st.rerun()

        st.divider()
        st.markdown("### 连接状态")
        online = bool(stats)
        state_class = "online" if online else ""
        state_text = "后端已连接" if online else "等待后端"
        st.markdown(
            f'<div class="sidebar-status"><span class="status-dot {state_class}"></span> {state_text}<br><span style="opacity:.7">{html.escape(API_BASE)}</span></div>',
            unsafe_allow_html=True,
        )
        if not API_TOKEN:
            st.warning("未读取到 API_BEARER_TOKEN，请确认后端启动后重新打开前端。")

        st.divider()
        with st.expander("隐私与使用边界"):
            privacy = api_get("/api/v1/privacy", show_error=False) or {}
            retention = privacy.get("image_retention_hours", CHAT_UPLOAD_RETENTION_HOURS)
            st.caption(
                f"皮肤照片仅用于当前分析，默认最多保留 {retention} 小时；"
                "上传前请确认照片属于你本人或你已获得照片本人的授权。"
            )
            st.caption(
                "聊天文本和皮肤图片可能发送给配置的外部模型服务。"
                "本工具只提供一般护肤信息，不是医疗诊断。"
            )

        st.markdown("### 使用提示")
        st.markdown(
            '<div class="sidebar-note">上传资料后点击“重建全部索引”。你可以问成分、产品、肤质、护肤流程，也可以上传皮肤照片让 Agent 做非医疗性的初步分析。</div>',
            unsafe_allow_html=True,
        )
    return stats


def _render_topbar(online: bool) -> None:
    state_class = "online" if online else ""
    state_text = "知识库在线" if online else "后端未连接"
    st.markdown(
        f'<div class="topbar"><div><div class="eyebrow">PERSONAL BEAUTY INTELLIGENCE</div><div class="page-title">你的专属美妆护肤顾问</div><div class="page-subtitle">从成分理解，到产品选择，再到更适合你的护理方案。</div></div><div class="status-pill"><span class="status-dot {state_class}"></span>{state_text}</div></div>',
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    mode = st.session_state.mode_selector
    description = "Agent 会按需搜索知识库、匹配图片，并把多个信息源整理成一份更完整的建议。" if mode == "Agent 模式" else "RAG 会从你的专属知识库中检索依据，再生成简洁、可执行的回答。"
    st.markdown(
        f'<div class="hero"><div class="hero-kicker">BEAUTY KNOWLEDGE, MADE PERSONAL</div><div class="hero-title">今天想了解什么？让专业知识变得更好用。</div><div class="hero-copy">{description}</div><div class="feature-row"><span class="feature-chip">✦ 成分百科</span><span class="feature-chip">✦ 产品匹配</span><span class="feature-chip">✦ 护肤流程</span><span class="feature-chip">✦ 图片分析</span></div></div>',
        unsafe_allow_html=True,
    )


def _render_empty_state() -> None:
    st.markdown(
        '<div class="empty-state"><div class="empty-icon">✦</div><div class="empty-title">从一个小问题开始</div><div class="empty-copy">你可以问我成分功效、产品区别、护肤顺序或小红书文案。我会基于当前知识库给出有依据的建议。</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">试试这些问题</div>', unsafe_allow_html=True)
    prompts = [
        ("成分解读", "烟酰胺和视黄醇可以一起用吗？"),
        ("产品选择", "帮我推荐适合油痘肌的保湿产品"),
        ("护肤流程", "请给我一套早C晚A的入门流程"),
    ]
    cols = st.columns(3)
    for col, (title, prompt) in zip(cols, prompts):
        with col:
            st.markdown(
                f'<div class="quick-card"><strong>{html.escape(title)}</strong>{html.escape(prompt)}</div>',
                unsafe_allow_html=True,
            )
            if st.button("使用这个问题", key=f"prompt_{title}", use_container_width=True):
                st.session_state.pending_query = prompt
                st.rerun()


def _render_images(images: list[dict[str, Any]]) -> None:
    if not images:
        return
    st.markdown('<div class="section-label">相关图片</div>', unsafe_allow_html=True)
    cols = st.columns(min(4, len(images)))
    for col, image in zip(cols, images[:4]):
        filename = str(image.get("filename", "图片"))
        path = str(image.get("path", ""))
        with col:
            if path.startswith("/static/images/"):
                st.image(f"{API_BASE}{path}", use_container_width=True)
            st.markdown(
                f'<div class="image-caption">{html.escape(filename)} · 匹配 {float(image.get("score", 0)):.3f}</div>',
                unsafe_allow_html=True,
            )


def _render_steps(steps: list[dict[str, Any]], expanded: bool = False) -> None:
    if not steps:
        return
    with st.expander("查看 Agent 工作过程", expanded=expanded):
        for step in steps:
            tool = html.escape(str(step.get("tool", "工具")))
            summary = html.escape(str(step.get("result_summary", "已完成")))
            st.markdown(
                f'<div class="agent-step"><strong>✦ {tool}</strong><br>{summary}</div>',
                unsafe_allow_html=True,
            )


def _render_message_extras(message: dict[str, Any]) -> None:
    extra = message.get("extra", {}) or {}
    safety_notice = extra.get("safety_notice")
    if safety_notice:
        st.warning(safety_notice)
    image_file_id = extra.get("image_file_id")
    if image_file_id:
        image_bytes = _chat_image_bytes(image_file_id)
        if image_bytes:
            st.image(image_bytes, width=120)
    _render_steps(message.get("steps", []))
    sources = extra.get("context_sources", []) or []
    if sources:
        tags = "".join(
            f'<span class="source-tag">▣ {html.escape(Path(str(source)).name)}</span>'
            for source in sources[:5]
        )
        st.markdown(tags, unsafe_allow_html=True)
    _render_images(extra.get("images", []) or [])


def _render_history() -> None:
    if not st.session_state.messages:
        _render_empty_state()
        return
    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        with st.chat_message(role, avatar="✦" if role == "assistant" else "你"):
            st.markdown(message.get("content", ""))
            _render_message_extras(message)


def _upload_chat_image() -> None:
    consented = st.checkbox(
        "我已阅读隐私说明，并同意将本人或已获授权的照片用于本次 AI 分析",
        key="privacy_consent",
    )
    if not consented:
        st.caption("上传前需要确认照片授权，以及图片可能发送至外部模型服务。")
        return

    uploaded = st.file_uploader(
        "上传皮肤照片",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
        key="chat_image_uploader",
    )
    if uploaded and uploaded.name != st.session_state.get("last_chat_upload_name"):
        st.session_state.last_chat_upload_name = uploaded.name
        with st.spinner("正在准备图片…"):
            result = api_post(
                "/api/v1/chat/upload_image",
                files={"file": (uploaded.name, uploaded.getvalue())},
            )
        if result:
            st.session_state.chat_image = result
            st.rerun()


def _render_attachment() -> None:
    image = st.session_state.chat_image
    if not image:
        return
    file_id = str(image.get("file_id", ""))
    name = str(image.get("filename", "图片"))
    image_bytes = _chat_image_bytes(file_id)
    if image_bytes:
        st.image(image_bytes, width=120)
    st.caption(f"已附加：{name}")
    if st.button("移除图片", key="remove_chat_image", type="secondary"):
        _delete_chat_image(file_id)
        st.session_state.chat_image = None
        st.session_state.last_chat_upload_name = None
        st.rerun()


def _history_payload() -> list[dict[str, str]]:
    history = []
    for message in st.session_state.messages[-8:]:
        role = message.get("role")
        content = message.get("content", "")
        if role in {"user", "assistant"} and content:
            history.append({"role": role, "content": str(content)[:4000]})
    return history


def _handle_query(query: str) -> None:
    image = st.session_state.chat_image
    image_filename = image.get("file_id") if image else None
    history = _history_payload()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
            "extra": {"image_file_id": image_filename} if image_filename else {},
        }
    )
    st.session_state.chat_image = None

    with st.chat_message("user", avatar="你"):
        st.markdown(query)
        image_bytes = _chat_image_bytes(image_filename)
        if image_bytes:
            st.image(image_bytes, width=120)

    with st.chat_message("assistant", avatar="✦"):
        answer_placeholder = st.empty()
        step_placeholder = st.empty()
        answer_parts: list[str] = []
        steps: list[dict[str, Any]] = []
        images: list[dict[str, Any]] = []
        sources: list[str] = []
        safety_notice = ""

        is_agent = st.session_state.mode_selector == "Agent 模式"
        payload: dict[str, Any] = {"question": query, "history": history}
        if image_filename:
            payload["image_filename"] = image_filename

        endpoint = "/api/v1/agent/chat/stream" if is_agent else "/api/v1/rag/chat/stream"
        for event in stream_events(endpoint, payload, timeout=150 if is_agent else 90):
            event_type = event.get("type")
            if event_type == "step":
                steps.append(
                    {
                        "tool": event.get("tool", "工具"),
                        "result_summary": "正在执行…",
                    }
                )
                step_placeholder.markdown(
                    "\n".join(f"`✦ {s['tool']}` · {s['result_summary']}" for s in steps)
                )
            elif event_type == "step_result":
                tool = event.get("tool", "工具")
                matched = next((s for s in reversed(steps) if s["tool"] == tool and s["result_summary"] == "正在执行…"), None)
                if matched:
                    matched["result_summary"] = event.get("summary", "已完成")
                else:
                    steps.append({"tool": tool, "result_summary": event.get("summary", "已完成")})
                step_placeholder.markdown(
                    "\n".join(f"`✦ {s['tool']}` · {s['result_summary']}" for s in steps)
                )
            elif event_type == "images":
                images = event.get("images", []) or []
                _render_images(images)
            elif event_type == "metadata":
                images = event.get("images", []) or []
                sources = event.get("sources", []) or []
            elif event_type == "safety":
                safety_notice = str(event.get("content", ""))
                if safety_notice:
                    st.warning(safety_notice)
            elif event_type == "text":
                answer_parts.append(str(event.get("content", "")))
                answer_placeholder.markdown("".join(answer_parts))
            elif event_type == "error":
                answer_parts.append(f"\n\n⚠️ {event.get('content', '请求失败')}")
                answer_placeholder.markdown("".join(answer_parts))

        step_placeholder.empty()
        final_answer = "".join(answer_parts).strip() or "暂时没有生成回答。"
        answer_placeholder.markdown(final_answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": final_answer,
                "steps": steps,
                "extra": {
                    "images": images,
                    "context_sources": sources,
                    "safety_notice": safety_notice,
                },
            }
        )
    st.rerun()


def _render_chat() -> None:
    _render_history()
    st.markdown("<div style='height:.35rem'></div>", unsafe_allow_html=True)
    _render_attachment()
    _upload_chat_image()

    pending = st.session_state.pending_query
    st.session_state.pending_query = None
    query = st.chat_input("问成分、产品、护肤流程，或上传照片让 Agent 分析…")
    if query is None and pending:
        query = pending

    if st.session_state.messages:
        col1, col2 = st.columns([5, 1])
        with col2:
            if st.button("清空对话", type="secondary", use_container_width=True):
                _delete_session_images()
                st.session_state.messages = []
                st.session_state.chat_image = None
                st.session_state.last_chat_upload_name = None
                st.rerun()

    if query and query.strip():
        _handle_query(query.strip())


def main() -> None:
    _init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    stats = _render_sidebar()
    _render_topbar(bool(stats))
    _render_hero()
    _render_chat()


if __name__ == "__main__":
    main()
