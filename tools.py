"""工具函数模块：提供厨房管家助手的辅助函数"""

import os
import json


def check_ingredients(ingredient_name: str, inventory: dict) -> dict:
    """
    检查冰箱中是否有指定食材
    参数：
        ingredient_name: 食材名称
        inventory: 当前库存字典
    返回：
        包含是否存在、数量、单位的字典
    """
    ingredient_name = ingredient_name.lower()
    for name, info in inventory.items():
        if ingredient_name in name.lower():
            return {
                "exists": True,
                "name": name,
                "quantity": info.get("quantity", 0),
                "unit": info.get("unit", "个"),
                "expiry_date": info.get("expiry_date", "未知")
            }
    return {"exists": False, "name": ingredient_name, "quantity": 0}
def generate_shopping_list(inventory: dict, target_dishes: list, recipes: dict) -> list:
    """
    根据当前库存和想要做的菜品，生成需要购买的食材清单
    参数：
        inventory: 当前库存字典
        target_dishes: 想要做的菜品列表
        recipes: 菜谱字典
    返回：
        需要购买的食材清单
    """
    shopping_list = {}
    for dish in target_dishes:
        if dish in recipes:
            required = recipes[dish].get("ingredients", {})
            for ingredient, needed_qty in required.items():
                # 检查库存
                current_qty = 0
                if ingredient in inventory:
                    current_qty = inventory[ingredient].get("quantity", 0)

                if current_qty < needed_qty:
                    need_qty = needed_qty - current_qty
                    if ingredient in shopping_list:
                        shopping_list[ingredient] = max(shopping_list[ingredient], need_qty)
                    else:
                        shopping_list[ingredient] = need_qty
    # 转换为列表格式
    result = []
    for ingredient, qty in shopping_list.items():
        result.append({
            "ingredient": ingredient,
            "quantity": qty,
            "unit": "个" if "蛋" in ingredient else "克" if "粉" in ingredient else "个"
        })
    return result

def calculate_recipe_feasibility(recipe_name: str, inventory: dict, recipe: dict) -> dict:
    """
    计算在当前库存下做某道菜的可行性
    参数：
        recipe_name: 菜名
        inventory: 当前库存字典
        recipe: 菜谱信息
    返回：
        可行性分析字典
    """
    required = recipe.get("ingredients", {})
    missing = []
    available = []
    for ingredient, needed in required.items():
        current = 0
        if ingredient in inventory:
            current = inventory[ingredient].get("quantity", 0)
        if current >= needed:
            available.append(ingredient)
        else:
            missing.append({
                "ingredient": ingredient,
                "needed": needed,
                "current": current,
                "shortage": needed - current
            })
    if len(missing) == 0:
        feasibility = 100
        suggestion = "食材齐全，可以直接做！"
    else:
        feasibility = int((len(available) / len(required)) * 100)
        suggestion = f"缺少{len(missing)}种食材：{', '.join([m['ingredient'] for m in missing])}"
    return {
        "recipe": recipe_name,
        "feasibility": feasibility,
        "available": available,
        "missing": missing,
        "suggestion": suggestion
    }

def load_recipes() -> dict:
    """加载菜谱数据"""
    return {
        "番茄炒蛋": {
            "ingredients": {"西红柿": 2, "鸡蛋": 3},
            "difficulty": "简单",
            "time": 15,
            "tags": ["家常", "快手"],
            "season": "四季"
        },
        "青椒肉丝": {
            "ingredients": {"青椒": 2, "猪肉": 200, "大蒜": 2},
            "difficulty": "中等",
            "time": 20,
            "tags": ["下饭"],
            "season": "四季"
        },
        "酸辣土豆丝": {
            "ingredients": {"土豆": 2, "干辣椒": 3, "醋": 10},
            "difficulty": "简单",
            "time": 15,
            "tags": ["快手", "酸辣"],
            "season": "四季"
        },
        "红烧排骨": {
            "ingredients": {"排骨": 500, "生姜": 10, "酱油": 20, "冰糖": 15},
            "difficulty": "困难",
            "time": 60,
            "tags": ["硬菜"],
            "season": "秋冬"
        },
        "凉拌黄瓜": {
            "ingredients": {"黄瓜": 2, "大蒜": 3, "醋": 10, "香油": 5},
            "difficulty": "简单",
            "time": 10,
            "tags": ["凉菜", "清爽"],
            "season": "夏季"
        }
    }