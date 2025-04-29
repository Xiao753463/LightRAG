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
            "cosine_better_than_threshold": 0.4  # Your desired threshold
        },
    )
# 在主程式結尾加上這段來正確執行 async 函數
if __name__ == "__main__":
    rag.auto_merge_similar_entities(
        similarity_threshold = 0.97,
        top_k=6,
        merge_strategy={
            "description": "concatenate",  # Combine all descriptions
            "source_id": "join_unique",     # Combine source IDs,  
            "mention_count": "add",     # Combine mention_count,  
            "sentiment_score": "add"     # Combine sentiment_score
        }
    )
