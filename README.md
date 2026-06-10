# Gemini-style AI Agent

基于 FastAPI 的 AI Agent，前端模仿 Google Gemini 的简洁暗色风格。

## 功能

- 💬 **对话模式** - 直接与 AI 对话
- 🔍 **联网搜索** - 通过 Tavily API 搜索互联网，基于搜索结果回答
- 📚 **知识库** - 从内部 Elasticsearch 知识库检索信息

## 依赖

```bash
pip install fastapi uvicorn httpx elasticsearch qwen-agent
```

## 启动

```bash
cd /home/azureuser/CASE-AI/gemini-agent
python app.py
```

服务运行在 `http://0.0.0.0:7860`

## 架构

```
gemini-agent/
├── app.py              # FastAPI 后端（LLM + Tavily + ES）
├── static/
│   └── index.html      # 前端单页面（Gemini 风格）
└── README.md
```

## API

- `POST /api/chat` - 流式对话（SSE），支持 mode: chat/search/knowledge
- `POST /api/search` - Tavily 搜索
- `POST /api/knowledge` - ES 知识库检索
