import asyncio
import nest_asyncio
import numpy as np

nest_asyncio.apply()
import os
import inspect
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
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
            "cosine_better_than_threshold": 0.2  # Your desired threshold
        },
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())


    # # Test different query modes
    # print("\nNaive Search:")
    # print(
    #     rag.query(
    #         "評論對象中Sentiment比例如何?", param=QueryParam(mode="naive")
    #     )
    # )

    print("\nLocal Search:")
    print(
        rag.query(
            "評論對象中Sentiment比例如何?", param=QueryParam(mode="local")
        )
    )

    # print("\nGlobal Search:")
    # print(
    #     rag.query(
    #         "消費者對於商品的看法如何?", param=QueryParam(mode="global")
    #     )
    # )

    # print("\nHybrid Search:")
    # print(
    #     rag.query(
    #         "消費者對於商品的看法如何?", param=QueryParam(mode="hybrid")
    #     )
    # )

    # # stream response
    # resp = rag.query(
    #     "消費者對於商品的看法如何?",
    #     param=QueryParam(mode="hybrid", stream=True),
    # )

    # if inspect.isasyncgen(resp):
    #     asyncio.run(print_stream(resp))
    # else:
    #     print(resp)


if __name__ == "__main__":
    main()
