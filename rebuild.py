"""独立重建文本和图片索引，供手动维护知识库时使用。"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

os.environ["MILVUS_URI"] = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")

print("正在连接 Milvus 数据库...")

from pymilvus import MilvusClient


async def run() -> None:
    # rebuild_* 复用 main.py 的全局客户端，因此这里显式初始化它。
    import main

    client = MilvusClient(uri=os.environ["MILVUS_URI"])
    main.milvus_client = client
    try:
        print("\n[1/2] 重建文本索引...")
        await main.rebuild_text_index()
        print("  ✅ 文本索引重建完成\n")

        print("[2/2] 重建图片索引...")
        await main.rebuild_image_index()
        print("  ✅ 图片索引重建完成\n")

        client.flush()
        if client.has_collection(main.settings.TEXT_COLLECTION):
            stats = client.get_collection_stats(main.settings.TEXT_COLLECTION)
            print(f'📄 文本切片: {stats.get("row_count", 0)}')
        if client.has_collection(main.settings.IMAGE_COLLECTION):
            stats = client.get_collection_stats(main.settings.IMAGE_COLLECTION)
            print(f'🖼️ 图片素材: {stats.get("row_count", 0)}')
        print("\n✅ 全部完成！请刷新前端页面 http://localhost:8501")
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(run())
