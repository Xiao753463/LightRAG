from datetime import datetime
import os
import numpy as np
import json
from collections import defaultdict

from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer

WORKING_DIR = "./0414"
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
        "cosine_better_than_threshold": 0.4
    },
)

with open("reviews.json", "r", encoding="utf-8") as f:
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
    date_key = datetime.strptime(item["date"], '%Y-%m-%d').date()
    date_string = date_key.strftime('%Y-%m-%d')
    source_id = f"review-{date_string.replace('-', '')}-{idx}"

    custom_kg["chunks"].append(
        {"content": review_text, "source_id": source_id})

    for analysis in analysis_items:
        topic = analysis["主題"]
        target = analysis["對象"].strip()
        facet = analysis["構面"]
        sentiment = analysis["情感"]

        sentiment_score = 1 if sentiment == "Positive" else - \
            1 if sentiment == "Negative" else 0
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

        facet_info = entity_info_map[facet_node_id][date_key]
        facet_info["count"] += 1
        facet_info["type"] = "構面"
        facet_info["desc"] = f"{brand} 在 {facet} 構面的評價。" if not target or target == "整體" else f"{brand} 的 {target} 在 {facet} 構面的評價。"
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
            obj_info["source_ids"].add(source_id)

            custom_kg["relationships"].extend([
                {"src_id": brand, "tgt_id": type_node_id,
                    "description": f"{brand} 提供的 {topic}。", "keywords": "提供", **edge_properties},
                {"src_id": type_node_id, "tgt_id": object_node_id,
                    "description": f"{topic} 包含 {target}。", "keywords": "包含", **edge_properties},
                {"src_id": object_node_id, "tgt_id": facet_node_id,
                    "description": f"{target} 具有 {facet} 的特性。", "keywords": "具有", **edge_properties},
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
