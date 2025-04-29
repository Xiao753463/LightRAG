from lightrag import QueryParam
from datetime import datetime
import os

import numpy as np
import pandas as pd
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
import asyncio
from sentence_transformers import SentenceTransformer
WORKING_DIR = "./0324"
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
            "cosine_better_than_threshold": 0.2  # Your desired threshold
        },
    )
query = QueryParam(
    entity_type="需求層級",
    entity_name="生理",
    with_neighbors=True,
    max_depth=2
)
result = rag.query(query)

# 從關聯節點中找出所有出現的 source_id
source_ids = set()
for node in result["nodes"]:
    if "source_id" in node:
        source_ids.add(node["source_id"])

print(f"共有 {len(source_ids)} 則評論與生理需求有關")