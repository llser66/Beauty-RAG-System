"""
一键重建索引 — 双击运行
直接在 Python 进程中执行，保证数据落库
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

os.environ["MILVUS_URI"] = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")

print("正在连接 Milvus 数据库...")

# 直接使用 main.py 中的函数
from main import rebuild_text_index, rebuild_image_index

print("\n[1/2] 重建文本索引...")
try:
    rebuild_text_index()
    print("  ✅ 文本索引重建完成\n")
except Exception as e:
    print(f"  ❌ 文本索引失败: {e}\n")

print("[2/2] 重建图片索引...")
try:
    rebuild_image_index()
    print("  ✅ 图片索引重建完成\n")
except Exception as e:
    print(f"  ❌ 图片索引失败: {e}\n")

# 验证
from pymilvus import MilvusClient, connections
try:
    connections.connect(alias="check_conn", uri=os.environ["MILVUS_URI"])
    cc = MilvusClient(alias="check_conn")
    cc.flush()

    if cc.has_collection("text_chunks"):
        s = cc.get_collection_stats("text_chunks")
        print(f'📄 文本切片: {s.get("row_count", 0)}')
    if cc.has_collection("image_embeddings"):
        s = cc.get_collection_stats("image_embeddings")
        print(f'🖼️ 图片素材: {s.get("row_count", 0)}')
    cc.close()
    connections.disconnect("check_conn")
except Exception as e:
    print(f"验证异常: {e}")

print("\n✅ 全部完成！请刷新前端页面 http://localhost:8501")
input("\n按 Enter 退出...")
