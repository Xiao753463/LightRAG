from sentence_transformers import SentenceTransformer
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag import LightRAG, QueryParam
import inspect
import os
import asyncio
import nest_asyncio
import numpy as np

nest_asyncio.apply()
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


async def initialize_rag():
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
            "cosine_better_than_threshold": 0.7  # Your desired threshold
        },
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)


# 加入多輪對話管理器
class ConversationManager:
    def __init__(self, history_limit=5):
        self.history = []
        self.history_limit = history_limit

    def add_turn(self, role, content):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit:]

    def get_history(self):
        return self.history

    def clear(self):
        self.history = []


def main():

    rag = asyncio.run(initialize_rag())
    conv = ConversationManager(history_limit=5)

    print("📣 輸入 'exit' 離開對話")
    while True:
        user_input = input("👤 你：")
        if user_input.lower() in ["exit", "quit"]:
            break

        conv.add_turn("user", user_input)

        response = rag.query(
            user_input,
            param=QueryParam(
                mode="local_plus",  # 你可改為 local / global / mix
                conversation_history=conv.get_history(),
                history_turns=5,
                top_k=20
            ),
        )

        if inspect.isasyncgen(response):
            print("🤖 AI（串流模式）：\n", end="", flush=True)
            asyncio.run(print_stream(response))
            print()
        else:
            print("🤖 AI：\n", response)
            conv.add_turn("assistant", response)


if __name__ == "__main__":
    main()
