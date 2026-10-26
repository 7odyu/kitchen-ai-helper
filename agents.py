"""厨房管家助手 - 多智能体核心模块"""

import os

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import datetime
import numpy as np
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
import chromadb
from tools import load_recipes

# ---------------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------------
short_term_memory: list[dict] = []  # 短期记忆，最多10轮
current_inventory: dict = {}  # 当前冰箱库存
dialog_count: int = 0  # 对话总轮数
user_name: str = ""  # 专门存储用户名字

# 默认库存
DEFAULT_INVENTORY = {
    "西红柿": {"quantity": 3, "unit": "个", "expiry_date": "2026-06-15"},
    "鸡蛋": {"quantity": 6, "unit": "个", "expiry_date": "2026-06-20"},
    "青椒": {"quantity": 2, "unit": "个", "expiry_date": "2026-06-12"},
    "土豆": {"quantity": 4, "unit": "个", "expiry_date": "2026-06-25"},
    "猪肉": {"quantity": 300, "unit": "克", "expiry_date": "2026-06-10"},
    "大蒜": {"quantity": 5, "unit": "瓣", "expiry_date": "2026-07-01"},
    "黄瓜": {"quantity": 2, "unit": "个", "expiry_date": "2026-06-14"},
}

current_inventory = DEFAULT_INVENTORY.copy()


# ---------------------------------------------------------------------------
# 1. BGE 嵌入模型加载
# ---------------------------------------------------------------------------
def load_bge_model() -> SentenceTransformer:
    """加载 BAAI/bge-large-zh-v1.5 嵌入模型"""
    model_name = "BAAI/bge-large-zh-v1.5"
    local_dir = "./models"
    model_path = os.path.join(local_dir, model_name.replace("/", os.sep))

    try:
        print(f"[模型] 尝试从本地路径加载: {model_path} ...")
        model = SentenceTransformer(model_path)
        print("[模型] 加载成功")
        return model
    except Exception:
        try:
            print(f"[模型] 从 ModelScope 下载...")
            from modelscope import snapshot_download
            snapshot_download(model_name, cache_dir=local_dir)
            model = SentenceTransformer(model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"BGE 模型加载失败: {e}")


print("[初始化] 加载 BGE 嵌入模型 ...")
bge_model = load_bge_model()
print("[初始化] BGE 模型就绪")


# ---------------------------------------------------------------------------
# 2. Chroma 向量数据库初始化
# ---------------------------------------------------------------------------
class BgeEmbeddingFunction:
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def __call__(self, input: list[str]) -> list[list[float]]:
        emb = self.model.encode(input)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb.tolist()

    def name(self) -> str:
        return "BAAI/bge-large-zh-v1.5"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.__call__(texts)


print("[初始化] 初始化 Chroma 向量数据库 ...")
chroma_client = chromadb.PersistentClient(path="./chroma_db_kitchen")
embedding_func = BgeEmbeddingFunction(bge_model)
memory_collection = chroma_client.get_or_create_collection(
    name="kitchen_memory",
    embedding_function=embedding_func
)
print("[初始化] Chroma 就绪")

# ---------------------------------------------------------------------------
# 3. 讯飞星火客户端
# ---------------------------------------------------------------------------
XUNFEI_API_KEY = "QYWdXAwRQJmMNCzNMcnH:kpVoLyNIGqSQUmSPldSB"

client = AsyncOpenAI(
    api_key=XUNFEI_API_KEY,
    base_url="https://spark-api-open.xf-yun.com/v1"
)
SPARK_MODEL = "lite"


# ---------------------------------------------------------------------------
# 智能体一：食材感知器
# ---------------------------------------------------------------------------
async def detect_ingredients(user_input: str) -> dict:
    """识别用户输入的食材信息"""
    global current_inventory

    prompt = f"""你是一个食材识别助手。分析以下用户输入，提取食材信息。
返回JSON格式：{{"ingredients": [{{"name":"食材名","quantity":数量,"action":"add"}}]}}
只返回JSON，不要其他内容。

用户输入：{user_input}"""
    try:
        resp = await client.chat.completions.create(
            model=SPARK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        text = resp.choices[0].message.content.strip()

        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        for ing in result.get("ingredients", []):
            name = ing.get("name", "")
            qty = ing.get("quantity", 1)
            action = ing.get("action", "add")
            if name:
                if action == "add":
                    if name in current_inventory:
                        current_inventory[name]["quantity"] += qty
                    else:
                        unit = "克" if ("肉" in name or "排骨" in name) else "个"
                        current_inventory[name] = {"quantity": qty, "unit": unit}
                    print(f"[库存] 添加 {name}: +{qty}")
                elif action == "remove":
                    if name in current_inventory:
                        current_inventory[name]["quantity"] = max(0, current_inventory[name]["quantity"] - qty)
                        print(f"[库存] 减少 {name}: -{qty}")
        return result
    except Exception as e:
        print(f"[错误] 食材识别失败: {e}")
        return {"ingredients": []}


def get_inventory_status() -> str:
    """获取当前库存状态文本"""
    if not current_inventory:
        return "冰箱里什么都没有"
    items = []
    for name, info in current_inventory.items():
        qty = info.get('quantity', 0)
        try:
            qty = int(qty)
        except (ValueError, TypeError):
            qty = 0
        if qty > 0:
            unit = info.get('unit', '个')
            items.append(f"{name}{qty}{unit}")
    if not items:
        return "冰箱里什么都没有"
    return "、".join(items)


# ---------------------------------------------------------------------------
# 智能体二：记忆管家（专门支持名字记忆）
# ---------------------------------------------------------------------------
async def extract_name(user_input: str) -> str:
    """专门提取用户名字"""
    global user_name
    prompt = f"""从用户输入中提取名字。用户说"我叫XXX"或"我是XXX"时，XXX就是名字。
只返回名字本身，不要返回其他内容。如果没有名字，返回"无"。
用户输入：{user_input}"""
    try:
        resp = await client.chat.completions.create(
            model=SPARK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=20,
        )
        name = resp.choices[0].message.content.strip()
        if name and name != "无" and len(name) <= 10:
            user_name = name
            # 同时存入数据库
            store_name_to_db(name)
            print(f"[名字记忆] 已记住用户名字: {name}")
            return name
    except:
        pass
    return ""

def store_name_to_db(name: str) -> None:
    """将名字存入数据库"""
    now = datetime.datetime.now()
    doc_id = f"name_{int(now.timestamp() * 1000)}"
    memory_collection.add(
        documents=[f"用户名字是{name}"],
        metadatas=[{"importance": 1.0, "timestamp": now.isoformat(), "type": "name"}],
        ids=[doc_id],
    )


def get_user_name() -> str:
    """从数据库获取用户名字"""
    global user_name
    if user_name:
        return user_name
    try:
        results = memory_collection.query(query_texts=["名字"], n_results=1)
        if results and results.get("documents") and results["documents"]:
            docs = results["documents"][0]
            for doc in docs:
                if "用户名字是" in doc:
                    name = doc.replace("用户名字是", "").strip()
                    user_name = name
                    return name
    except:
        pass
    return ""


async def extract_preference(user_input: str, reply: str) -> list[dict]:
    """提取用户饮食偏好"""
    prompt = f"""从以下对话中提取用户的饮食偏好。
返回JSON列表：[{{"content":"偏好内容","importance":0.0~1.0}}]
只提取：不吃辣、喜欢甜、过敏等，不要提取名字。
用户：{user_input}
助手：{reply}"""
    try:
        resp = await client.chat.completions.create(
            model=SPARK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        text = resp.choices[0].message.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except:
        return []


def store_memory(info_list: list[dict]) -> None:
    """存入长期记忆"""
    now = datetime.datetime.now()
    for i, info in enumerate(info_list):
        content = info.get("content", "").strip()
        if not content:
            continue
        importance = min(1.0, max(0.0, float(info.get("importance", 0.5))))
        doc_id = f"mem_{int(now.timestamp() * 1000)}_{i}"
        memory_collection.add(
            documents=[content],
            metadatas=[{"importance": importance, "timestamp": now.isoformat()}],
            ids=[doc_id],
        )
        print(f"[记忆] 已存储: {content}")


def retrieve_memories(query: str, n: int = 3) -> str:
    """检索相关记忆"""
    try:
        count = memory_collection.count()
        if count == 0:
            return ""
        n = min(n, count)
        results = memory_collection.query(query_texts=[query], n_results=n)
        if results and results.get("documents") and results["documents"]:
            docs = results["documents"][0]
            return "\n".join(f"- {d}" for d in docs)
    except:
        pass
    return ""


# ---------------------------------------------------------------------------
# 智能体三：性格引擎
# ---------------------------------------------------------------------------
PERSONA_FILE = "kitchen_persona.json"
DEFAULT_PERSONA = {
    "热情程度": 0.7,
    "细致程度": 0.5,
    "效率优先程度": 0.4,
    "健康关注程度": 0.6,
    "互动总轮数": 0,
}

def load_persona() -> dict:
    if os.path.exists(PERSONA_FILE):
        with open(PERSONA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_PERSONA.copy()
def save_persona(params: dict) -> None:
    with open(PERSONA_FILE, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

async def update_personality(recent_dialogs: list[dict]) -> None:
    """更新性格参数"""
    params = load_persona()
    params["互动总轮数"] = params.get("互动总轮数", 0) + 10
    recent_text = " ".join(d.get("user", "") for d in recent_dialogs[-10:])
    changed = False
    if any(kw in recent_text for kw in ["谢谢", "真好", "太棒了"]):
        params["热情程度"] = min(1.0, params["热情程度"] + 0.05)
        changed = True
        print(f"[性格] 热情程度 +0.05 → {params['热情程度']:.2f}")
    if any(kw in recent_text for kw in ["具体", "详细", "怎么做"]):
        params["细致程度"] = min(1.0, params["细致程度"] + 0.04)
        changed = True
        print(f"[性格] 细致程度 +0.04 → {params['细致程度']:.2f}")
    if any(kw in recent_text for kw in ["赶时间", "快点", "简单"]):
        params["效率优先程度"] = min(1.0, params["效率优先程度"] + 0.05)
        changed = True
        print(f"[性格] 效率优先程度 +0.05 → {params['效率优先程度']:.2f}")
    if any(kw in recent_text for kw in ["减肥", "清淡", "健康"]):
        params["健康关注程度"] = min(1.0, params["健康关注程度"] + 0.06)
        changed = True
        print(f"[性格] 健康关注程度 +0.06 → {params['健康关注程度']:.2f}")

    for key in params:
        if isinstance(params[key], float):
            params[key] = max(0.0, min(1.0, params[key]))

    if changed:
        save_persona(params)


def get_persona_prompt(params: dict) -> str:
    parts = []
    if params.get("热情程度", 0.5) > 0.6:
        parts.append("热情活泼")
    if params.get("细致程度", 0.5) > 0.5:
        parts.append("细心周到")
    if params.get("效率优先程度", 0.5) > 0.5:
        parts.append("推荐快手菜")
    if params.get("健康关注程度", 0.5) > 0.6:
        parts.append("关注健康")
    return "，".join(parts) if parts else "贴心"


# ---------------------------------------------------------------------------
# 智能体四：对话精灵
# ---------------------------------------------------------------------------
# async def generate_reply(
#         user_input: str,
#         recent: list[dict],
#         memories: str,
#         inventory_status: str,
#         persona_prompt: str,
# ) -> str:
#     """生成最终回复"""
#
#     # 获取用户名字
#     name = get_user_name()
#
#     # 构建最近对话
#     recent_text = ""
#     for d in recent[-3:]:
#         recent_text += f"主人：{d['user']}\n助手：{d['bot']}\n"
#
#     # 获取菜谱
#     recipes = load_recipes()
#     recipe_names = "、".join(list(recipes.keys()))
#
#     # 构建 system prompt
#     if name:
#         system_prompt = f"""你是厨房管家「小厨」，{persona_prompt}。
#
# 【重要】主人的名字是{name}，每次回复必须称呼他！
#
# 冰箱里有：{inventory_status}
# 可用菜谱：{recipe_names}
# 用户偏好：{memories}
#
# 回复要求：
# - 用"{name}，"开头
# - 简短（不超过60字）
# - 推荐菜只说菜名
#
# 请回复：{user_input}"""
#     else:
#         system_prompt = f"""你是厨房管家「小厨」，{persona_prompt}。
#
# 冰箱里有：{inventory_status}
# 可用菜谱：{recipe_names}
# 用户偏好：{memories}
#
# 回复要求：
# - 简短（不超过60字）
# - 推荐菜只说菜名
#
# 请回复：{user_input}"""
#
#     print(f"[调试] 名字:{name if name else '未知'} 问题:{user_input}")
#
#     try:
#         resp = await client.chat.completions.create(
#             model=SPARK_MODEL,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_input},
#             ],
#             temperature=0.7,
#             max_tokens=150,
#         )
#         result = resp.choices[0].message.content.strip()
#         return result
#     except Exception as e:
#         print(f"[错误] {e}")
#         return f"冰箱里有{inventory_status}，想试试番茄炒蛋吗？"
async def generate_reply(
        user_input: str,
        recent: list[dict],
        memories: str,
        inventory_status: str,
        persona_prompt: str,
) -> str:
    """生成最终回复"""

    name = get_user_name()

    # 构建系统提示词
    if name:
        system_prompt = f"你是厨房管家「小厨」。主人的名字是{name}，每次回复用'{name}，'开头。"
    else:
        system_prompt = "你是厨房管家「小厨」，热情帮助主人解决厨房问题。"

    system_prompt += f" 冰箱里有：{inventory_status}。如果主人问的菜谱冰箱里没有的食材，你可以搜索网络提供菜谱。"

    try:
        resp = await client.chat.completions.create(
            model=SPARK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=0.7,
            max_tokens=500,
            # 关键：开启联网搜索
            extra_body={
                "tools": [{
                    "type": "web_search",
                    "web_search": {
                        "enable": True  # 开启联网搜索
                    }
                }]
             }
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[错误] {e}")
        return f"冰箱里有{inventory_status}，想试试番茄炒蛋吗？"