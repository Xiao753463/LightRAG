from datetime import datetime
import os
import numpy as np
import pandas as pd
from collections import defaultdict

from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer

WORKING_DIR = "./0330"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "753463"
os.environ["NEO4J_PASSWORD"] = "forfun963741"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

model = SentenceTransformer("intfloat/multilingual-e5-large")
async def embedding_func(texts: list[str]) -> np.ndarray:
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="llama3.1:8b-instruct-q5_K_M",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={
        "host": "http://localhost:11434",
        "options": {"num_ctx": 32768},
    },
    embedding_func=EmbeddingFunc(
        embedding_dim=model.get_sentence_embedding_dimension(),
        max_token_size=8192,
        func=embedding_func,
    ),
    graph_storage="Neo4JStorage",
    vector_storage="FaissVectorDBStorage",
    vector_db_storage_cls_kwargs={
        "cosine_better_than_threshold": 0.2
    },
)

# 讀取 CSV 檔案
file_path = "reviews.csv"  # 替換成你的檔案路徑
df = pd.read_csv(file_path)

custom_kg = {"chunks": [], "entities": [], "relationships": []}

# 加入計數與情感加總機制
entity_info_map = defaultdict(lambda: {
    "count": 0,
    "type": "",
    "desc": "",
    "sentiment_score": 0,
    "source_ids": set()
})

# 解析每一筆評論資料
for _, row in df.iterrows():
    brand = row["Brand"]
    store = row["Store"]
    review_text = row["Review"]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    source_id = f"review-{timestamp}-{_}"

    objects = row["Extract"].split("\n")
    topics = row["Topic"].split("\n")
    maslow_levels = row["Maslow_Level"].split("\n")
    sentiments = row["Sentiment"].split("\n")

    custom_kg["chunks"].append({"content": review_text, "source_id": source_id})

    for i in range(len(objects)):
        obj_info = objects[i].split(" ")
        if len(obj_info) < 2:
            continue

        obj_name = obj_info[0]
        obj_description = obj_info[1]
        topic = topics[i] if i < len(topics) else "未分類"
        maslow = maslow_levels[i] if i < len(maslow_levels) else "未知"
        sentiment = sentiments[i] if i < len(sentiments) else "未知"
        sentiment_score = 1 if sentiment == "Positive" else -1 if sentiment == "Negative" else 0
        # 累積屬性與情感分數
        for name, etype, desc in [
            (brand, "品牌", f"{brand} 是一家知名品牌。"),
            (store, "分店", f"{store} 是 {brand} 的分店。"),
            (obj_name, topic, obj_description),
            (topic, "主題", f"這則評論涉及 {topic}。"),
            (maslow, "需求層級", f"此內容與 {maslow} 需求相關。")
        ]:
            entity_info_map[name]["count"] += 1
            entity_info_map[name]["type"] = etype
            entity_info_map[name]["desc"] = desc
            entity_info_map[name]["source_ids"].add(source_id)
            entity_info_map[name]["sentiment_score"] += sentiment_score

        # 關係（去除情感節點）
        custom_kg["relationships"].extend([
            {"src_id": brand, "tgt_id": store,"description": f"{brand} 經營 {store}。","keywords": "經營", "weight": 1.0, "source_id": source_id},
            {"src_id": store, "tgt_id": topic, "description": f"{store} 提供 {topic} 。", "keywords": topic, "weight": 1.0, "source_id": source_id},
            {"src_id": topic, "tgt_id": obj_name, "description": f"{topic} 包含 {obj_name}。", "keywords": "包含", "weight": 1.0, "source_id": source_id},
            {"src_id": obj_name, "tgt_id": maslow, "description": f"{obj_name} 提供 {maslow} 需求。", "keywords": "提供", "weight": 1.0, "source_id": source_id}
        ])

print("----整合結果----")
# 將整理好的實體資訊加入圖譜中
for name, info in entity_info_map.items():
    print("實體: ", name, "情感: ", info["sentiment_score"])
    custom_kg["entities"].append({
        "entity_name": name,
        "entity_type": info["type"],
        "description": info["desc"],
        "mention_count": info["count"],
        "sentiment_score": info["sentiment_score"],  # ✅ 累積情感分數
        "source_id": "<SEP>".join(info["source_ids"])    # 多筆來源合併
    })
    print(custom_kg["entities"])

# 寫入圖譜
rag.insert_custom_kg(custom_kg)

