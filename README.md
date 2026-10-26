# 厨房小能手智能体 (Kitchen AI Helper)

## 项目简介
本项目是一个基于大语言模型（LLM）开发的多场景智能问答系统，核心功能包括厨房助手等角色的对话交互。

## 技术栈
- **后端：** Python, 大模型 API 接入
- **前端：** HTML
- **数据存储：** JSON 配置文件及本地向量库

## 核心功能
- 实现多智能体（Agent）的角色设定与对话逻辑。
- 结合上下文进行智能问答。
- 结构清晰的配置与代码分离（kitchen_persona.json 等）。

## 如何运行
1. 安装依赖：`pip install -r requirements.txt`
2. 配置你的大模型 API Key
3. 运行主程序：`python main.py`
