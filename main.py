import sys, asyncio, os, json
import traceback

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


class RAGClient:
    def __init__(self):
        self.session = None
        self.transport = None
        self.client = OpenAI(api_key="novo", base_url="http://127.0.0.1:8000/v1")

        self.tools = None

    async def connect(self, server_script: str):
        params = StdioServerParameters(
            command="uvx",
            args=["mcp-rag"],
        )

        self.transport = stdio_client(params)
        self.read_stream, self.write_stream = await self.transport.__aenter__()

        self.session = await ClientSession(
            self.read_stream, self.write_stream
        ).__aenter__()

        await self.session.initialize()

        resp = await self.session.list_tools()

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in resp.tools
        ]
        print("可用工具：", [t["function"]["name"] for t in self.tools])

    async def query(self, q: str):
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的医学助手，请根据提供的医学文档回答问题。如果用户的问题需要查询医学知识，请使用列表中的工具来获取相关信息。",
            },
            {"role": "user", "content": q},
        ]

        while True:
            try:
                response = self.client.chat.completions.create(
                    model="Qwen2.5-7B-Instruct-AWQ",
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                )
                message = response.choices[0].message
                messages.append(message)

                if not message.tool_calls:
                    return message.content

                for tool_call in message.tool_calls:
                    print("tool_call", tool_call)
                    args = json.loads(tool_call.function.arguments)
                    result = await self.session.call_tool(tool_call.function.name, args)
                    print("result", result)
                    messages.append(
                        {
                            "role": "tool",
                            "content": str(result),
                            "tool_call_id": tool_call.id,
                        }
                    )
            except Exception as e:
                print(f"发生错误: {str(e)}")
                traceback.print_exc()
                return "抱歉，处理您的请求时出现了问题。"

    async def close(self):
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
            if self.transport:
                await self.transport.__aexit__(None, None, None)
        except Exception as e:
            print(f"关闭连接时发生错误: {str(e)}")


async def main():
    print(">>> 开始初始化 RAG 系统")

    if len(sys.argv) < 2:
        print("用法: python main.py <the args of mcp server>")
        return

    client = RAGClient()

    await client.connect(sys.argv[1])

    print(">>> 系统链接成功")

    # 添加一些医学文档
    medical_docs = [
        "糖尿病是一种慢性代谢性疾病，主要特征是血糖水平持续升高。",
        "高血压是指动脉血压持续升高，通常定义为收缩压≥200mmHg和/或舒张压≥90mmHg。",
        "冠心病是由于冠状动脉粥样硬化导致心肌缺血缺氧的疾病。",
        "哮喘是一种慢性气道炎症性疾病，表现为反复发作的喘息、气促、胸闷和咳嗽。",
        "肺炎是由细菌、病毒或其他病原体引起的肺部感染，常见症状包括发热、咳嗽和呼吸困难。",
    ]

    print(">>> 正在索引医学文档...")

    res = await client.session.call_tool("index_docs", {"docs": medical_docs})

    print(">>> 文档索引完成")

    while True:
        print("\n请输入您要查询的医学问题（输入'退出'结束查询）：")
        query = input("> ")

        if query.lower() == "退出":
            break

        print(f"\n正在查询: {query}")

        response = await client.query(query)
        print("\nAI 回答：\n", response)

    await client.close()
    print(">>> 系统已关闭")


if __name__ == "__main__":
    asyncio.run(main())
