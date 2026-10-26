import asyncio
from openai import AsyncOpenAI

async def test():
    client = AsyncOpenAI(
        api_key="DAhKzbnlhDTLRmxWxoDM:QOdjtIvDxuekGZrHIQql",  # 替换成实际的
        base_url="https://spark-api-open.xf-yun.com/v1"
    )
    try:
        resp = await client.chat.completions.create(
            model="lite",
            messages=[{"role": "user", "content": "说你好"}],
            max_tokens=50
        )
        print("API 调用成功:", resp.choices[0].message.content)
    except Exception as e:
        print("API 调用失败:", e)

asyncio.run(test())