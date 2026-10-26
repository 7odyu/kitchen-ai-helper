"""厨房管家助手 - FastAPI应用入口"""

import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

import agents
from tools import load_recipes, calculate_recipe_feasibility, generate_shopping_list


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[启动] 厨房管家助手已启动！")
    yield
    print("[关闭] 厨房管家助手休息了……")


app = FastAPI(title="厨房管家助手", lifespan=lifespan)

# 挂载静态文件
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/status")
async def get_status():
    """返回当前状态"""
    persona = agents.load_persona()
    return JSONResponse({
        "inventory": agents.current_inventory,
        "persona_params": persona,
        "dialog_count": agents.dialog_count,
    })


@app.get("/inventory")
async def get_inventory():
    """获取库存信息"""
    return JSONResponse({
        "inventory": agents.current_inventory,
        "summary": agents.get_inventory_status()
    })


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return JSONResponse({
            "reply": "😋 主人想说什么？小厨听着呢！",
            "inventory": agents.get_inventory_status(),
        })

    # 新增：提取名字
    await agents.extract_name(user_input)

    # 1. 食材感知
    await agents.detect_ingredients(user_input)
    inventory_status = agents.get_inventory_status()

    # 2. 长期记忆检索
    memories = agents.retrieve_memories(user_input)

    # 3. 性格引擎
    persona_params = agents.load_persona()
    persona_prompt = agents.get_persona_prompt(persona_params)

    # 4. 生成回复
    reply = await agents.generate_reply(
        user_input,
        agents.short_term_memory,
        memories,
        inventory_status,
        persona_prompt,
    )

    # 5. 提取偏好并存储
    pref_list = await agents.extract_preference(user_input, reply)
    agents.store_memory(pref_list)

    # 6. 更新短期记忆
    agents.short_term_memory.append({"user": user_input, "bot": reply})
    while len(agents.short_term_memory) > 10:
        agents.short_term_memory.pop(0)

    # 7. 每10轮更新性格
    agents.dialog_count += 1
    if agents.dialog_count % 10 == 0:
        await agents.update_personality(agents.short_term_memory)

    return JSONResponse({
        "reply": reply,
        "inventory": inventory_status,
        "persona_params": persona_params,
        "dialog_count": agents.dialog_count,
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)