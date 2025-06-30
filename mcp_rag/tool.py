from typing import Any, List

import os
import faiss
import numpy as np
from openai import OpenAI, embeddings
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("rag")


class Tool:
    def __init__(self):
        self._index: faiss.IndexFlatL2 = faiss.IndexFlatL2(1536)
        self._docs: List[str] = []

        self._openai = OpenAI(api_key="imind", base_url="http://10.10.1.75:8000/v1")

    async def embed_text(self, texts: List[str]) -> np.ndarray:
        resp = self._openai.embeddings.create(
            model="Qwen2.5-7B-Instruct-AWQ", input=texts, encoding_format="float"
        )
        return np.array([d.embedding for d in resp.data], dtype="float32")

    async def index_docs(self, docs: List[str]) -> str:
        """将一批文档加入索引。
        Args:
            docs: 文本列表
        """

        embeddings = await self.embed_text(docs)
        self._index.add(embeddings.astype("float32"))
        self._docs.extend(docs)

        return f"已索引 {len(docs)} 篇文档，总文档数：{len(self._docs)}"

    async def retrieve_docs(self, query: str, top_k: int = 3) -> str:
        """检索最相关文档片段。
        Args:
            query: 用户查询
            top_k: 返回的文档数
        """
        q_emb = await self.embed_text([query])
        D, I = self._index.search(q_emb.astype("float32"), top_k)
        results = [f"[{i}] {self._docs[i]}" for i in I[0] if i < len(self._docs)]
        return "\n\n".join(results) if results else "未检索到相关文档。"


tool = Tool()

mcp.add_tool(tool.index_docs)
mcp.add_tool(tool.retrieve_docs)
