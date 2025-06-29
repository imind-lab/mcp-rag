from typing import Any, List

import os
import faiss
import numpy as np
from openai import OpenAI, embeddings
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("rag")

_index: faiss.IndexFlatL2 = faiss.IndexFlatL2(1536)
_docs: List[str] = []

openai = OpenAI(api_key="imind", base_url="http://127.0.0.1:8000/v1")


async def embed_text(texts: List[str]) -> np.ndarray:
    resp = openai.embeddings.create(
        model="Qwen2.5-7B-Instruct-AWQ", input=texts, encoding_format="float"
    )
    return np.array([d.embedding for d in resp.data], dtype="float32")


@mcp.tool()
async def index_docs(docs: List[str]) -> str:
    """将一批文档加入索引。
    Args:
        docs: 文本列表
    """
    global _index, _docs

    embeddings = await embed_text(docs)
    _index.add(embeddings.astype("float32"))
    _docs.extend(docs)

    return f"已索引 {len(docs)} 篇文档，总文档数：{len(_docs)}"


@mcp.tool()
async def retrieve_docs(query: str, top_k: int = 3) -> str:
    """检索最相关文档片段。
    Args:
        query: 用户查询
        top_k: 返回的文档数
    """
    q_emb = await embed_text([query])
    D, I = _index.search(q_emb.astype("float32"), top_k)
    results = [f"[{i}] {_docs[i]}" for i in I[0] if i < len(_docs)]
    return "\n\n".join(results) if results else "未检索到相关文档。"
