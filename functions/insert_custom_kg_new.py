from datetime import datetime
import os
import numpy as np
import json
from collections import defaultdict

from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer

WORKING_DIR = "./RAG"
# Neo4j 連線資訊請在環境變數或 .env 中設定：NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")

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
        "cosine_better_than_threshold": 0.4
    },
)

with open("reviews_test4.json", "r", encoding="utf-8") as f:
    reviews_data = json.load(f)

custom_kg = {"chunks": [], "entities": [], "relationships": []}
entity_info_map = defaultdict(lambda: defaultdict(lambda: {
    "count": 0,
    "source_ids": set(),
    "type": "",
    "desc": ""
}))

for idx, item in enumerate(reviews_data):
    brand = item["brand"]
    review_text = item["review"]
    analysis_items = item["analysis"]

    # 如果 analysis 是空列表，就跳過這筆
    if not analysis_items:
        continue
    print(f"分析評論：{analysis_items}")

    date_key = datetime.strptime(item["date"], '%Y-%m-%d').date()
    date_string = date_key.strftime('%Y-%m-%d')
    source_id = f"review-{date_string.replace('-', '')}-{idx}"

    # 檢查是否有句子切分
    if "sentences" in item and item["sentences"]:
        sentences = item["sentences"].split('\n')
        # 將所有句子作為一個chunk，使用同一個source_id
        for sentence in sentences:
            if sentence.strip():  # 排除空句子
                custom_kg["chunks"].append(
                    {"content": sentence.strip(), "source_id": source_id})
    else:
        # 如果沒有sentence欄位，就使用整段review作為chunk
        custom_kg["chunks"].append(
            {"content": review_text, "source_id": source_id})

    for analysis in analysis_items:
        topic = analysis["主題"]
        target = analysis["對象"].strip()
        facet = analysis["構面"]
        sentiment = analysis["情感"]
        # 獲取意見詞，如果沒有則為空字符串
        opinion = analysis.get("意見詞", "").strip()

        try:
            sentiment_score = float(sentiment)
        except (ValueError, TypeError):
            print(f"無法轉換情感值 '{sentiment}' 為浮點數，將其設為 0.0")
            sentiment_score = 0.0

        object_type = "商品" if topic == "商品" else "服務"

        type_node_id = f"{topic}"
        facet_node_id = f"{facet}"

        for level, node_id, etype, desc in [
            ("品牌", brand, "品牌", f"{brand} 是一家知名品牌。"),
            ("類型", type_node_id, "類型", f"{topic} 是一種評論對象類別。"),
        ]:
            info = entity_info_map[node_id][date_key]
            info["count"] += 1
            info["type"] = etype
            info["desc"] = desc
            info["source_ids"].add(source_id)

        # 更新構面的描述，加入意見詞
        facet_desc = ""
        if not target or target == "整體":
            facet_desc = f"{brand} 在 {facet} 構面的評價"
        else:
            facet_desc = f"{brand} 的 {target} 在 {facet} 構面的評價"

        # 如果有意見詞，加入到描述中
        if opinion:
            facet_desc += f"：{opinion}"
        else:
            facet_desc += "。"

        facet_info = entity_info_map[facet_node_id][date_key]
        facet_info["count"] += 1
        facet_info["type"] = "構面"
        facet_info["desc"] = facet_desc
        facet_info["source_ids"].add(source_id)

        edge_properties = {
            "brand": brand,
            "sentiment_score": sentiment_score,
            "source_id": source_id,
            "date": date_key
        }

        if target:
            object_node_id = f"{target}"
            obj_info = entity_info_map[object_node_id][date_key]
            obj_info["count"] += 1
            obj_info["type"] = topic
            obj_info["desc"] = f"{brand} 的 {target}"
            if opinion:
                obj_info["desc"] += f"，評價為：{opinion}"
            obj_info["source_ids"].add(source_id)

            # 建立關係時加入意見詞的資訊
            relationship_desc = f"{target} 具有 {facet} 的特性"
            if opinion:
                relationship_desc += f"，評價為：{opinion}"
            else:
                relationship_desc += "。"

            custom_kg["relationships"].extend([
                {"src_id": brand, "tgt_id": type_node_id,
                    "description": f"{brand} 提供的 {topic}。", "keywords": "提供", **edge_properties},
                {"src_id": type_node_id, "tgt_id": object_node_id,
                    "description": f"{topic} 包含 {target}。", "keywords": "包含", **edge_properties},
                {"src_id": object_node_id, "tgt_id": facet_node_id, "opinion": opinion,
                    "description": relationship_desc, "keywords": "具有", **edge_properties},
            ])

print(f"custom_kg['relationships']: {custom_kg['relationships']}")

for name, date_dict in entity_info_map.items():
    for date, info in date_dict.items():
        custom_kg["entities"].append({
            "entity_name": name,
            "entity_type": info["type"],
            "description": info["desc"],
            "mention_count": info["count"],
            "source_id": "<SEP>".join(info["source_ids"])
        })

print("開始插入知識圖譜與 chunks...")
rag.insert_custom_kg(custom_kg)
print("插入完成")
