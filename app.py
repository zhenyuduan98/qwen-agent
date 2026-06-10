"""
Gemini-style AI Agent - 纯粹模式
Agent 自主决定何时调搜索/知识库，前端展示思考链 + 工具调用
"""
import os
import sys
import json
import copy
import time
import uuid
import sqlite3
from typing import Generator
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

from qwen_agent.agents import Assistant

app = FastAPI()

# --- Config ---
LLM_CFG = {
    'model': 'claude-opus-4.8',
    'model_server': os.getenv('LLM_MODEL_SERVER', ''),
    'api_key': os.getenv('LLM_API_KEY', ''),
    'generate_cfg': {'top_p': 0.8}
}

RAG_CFG = {
    "rag_backend": "elasticsearch",
    "es": {
        "host": "https://localhost",
        "port": 9200,
        "user": "elastic",
        "password": os.getenv('ES_PASSWORD', ''),
        "index_name": "my_insurance_docs_index"
    },
    "parser_page_size": 500
}

TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', '')

SYSTEM_INSTRUCTION = '''你是一个智能AI助手。
你有以下能力：
1. 通过 tavily_search 工具搜索互联网，获取最新信息
2. 通过 retrieval 工具从本地知识库检索专业文档（保险条款等）

请根据用户的问题自行判断：
- 如果问题涉及本地知识库中的内容（如保险产品、条款），优先使用 retrieval 工具
- 如果问题需要最新的互联网信息，使用 tavily_search 工具
- 如果是一般性问题，直接回答即可

回答时请引用信息来源。'''

TOOLS_CFG = [{
    "mcpServers": {
        "tavily-mcp": {
            "command": "npx",
            "args": ["-y", "tavily-mcp@0.1.4"],
            "env": {
                "TAVILY_API_KEY": TAVILY_API_KEY
            },
            "disabled": False,
            "autoApprove": []
        }
    }
}]

# --- 获取知识库文件列表 ---
file_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs')
files = []
if os.path.exists(file_dir):
    for f in os.listdir(file_dir):
        fp = os.path.join(file_dir, f)
        if os.path.isfile(fp):
            files.append(fp)
print(f'知识库文件列表 ({len(files)} 个):', [os.path.basename(f) for f in files])

# --- 创建 Agent ---
bot = Assistant(
    llm=LLM_CFG,
    system_message=SYSTEM_INSTRUCTION,
    function_list=TOOLS_CFG,
    files=files,
    rag_cfg=RAG_CFG
)

# --- SQLite Database ---
DB_PATH = os.path.join(os.path.dirname(__file__), 'chat_history.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        title TEXT,
        created_at TEXT,
        updated_at TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content TEXT,
        created_at TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    )''')
    conn.commit()
    conn.close()

init_db()

def db_create_session(session_id: str, title: str = "新对话"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute('INSERT OR IGNORE INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)',
              (session_id, title, now, now))
    conn.commit()
    conn.close()

def db_update_session_title(session_id: str, title: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute('UPDATE sessions SET title=?, updated_at=? WHERE id=?', (title, now, session_id))
    conn.commit()
    conn.close()

def db_save_message(session_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute('INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)',
              (session_id, role, content, now))
    c.execute('UPDATE sessions SET updated_at=? WHERE id=?', (now, session_id))
    conn.commit()
    conn.close()

def db_get_sessions(limit: int = 30):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?', (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def db_get_messages(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT role, content, created_at FROM messages WHERE session_id=? ORDER BY id ASC', (session_id,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def db_delete_session(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM messages WHERE session_id=?', (session_id,))
    c.execute('DELETE FROM sessions WHERE id=?', (session_id,))
    conn.commit()
    conn.close()

# In-memory session cache (for bot context)
sessions = {}

# --- API Models ---
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(os.path.dirname(__file__), "static", "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/sessions")
async def list_sessions():
    """获取历史对话列表"""
    return JSONResponse(db_get_sessions())

@app.post("/api/sessions/new")
async def new_session():
    """创建新对话"""
    sid = str(uuid.uuid4())[:8]
    db_create_session(sid)
    return JSONResponse({"session_id": sid})

@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """获取某个对话的所有消息"""
    msgs = db_get_messages(session_id)
    return JSONResponse(msgs)

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除某个对话"""
    db_delete_session(session_id)
    if session_id in sessions:
        del sessions[session_id]
    return JSONResponse({"ok": True})

@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """Streaming chat with SSE - 包含思考链和工具调用信息"""
    session_id = req.session_id

    # Ensure session exists in DB
    db_create_session(session_id)

    # Load messages from DB if not in memory
    if session_id not in sessions:
        db_msgs = db_get_messages(session_id)
        sessions[session_id] = [{'role': m['role'], 'content': m['content']} for m in db_msgs]
    
    messages = sessions[session_id]
    messages.append({'role': 'user', 'content': req.message})

    # Save user message to DB
    db_save_message(session_id, 'user', req.message)

    # Update title with first user message
    if len([m for m in messages if m['role'] == 'user']) == 1:
        title = req.message[:30] + ('...' if len(req.message) > 30 else '')
        db_update_session_title(session_id, title)

    def generate():
        prev_resp_len = 0
        full_response_text = ""
        last_assistant_idx = -1
        seen_tool_calls = set()
        seen_tool_results = set()
        
        try:
            # 先查 ES 看是否有相关知识库内容
            has_rag_results = False
            try:
                import httpx
                es_query = {
                    "query": {"match": {"content": req.message}},
                    "size": 3,
                    "_source": ["file_name"]
                }
                es_resp = httpx.post(
                    f"https://localhost:9200/my_insurance_docs_index/_search",
                    json=es_query,
                    auth=("elastic", os.getenv('ES_PASSWORD', '')),
                    verify=False,
                    timeout=5
                )
                hits = es_resp.json().get('hits', {}).get('hits', [])
                # 只有分数超过阈值才算“有结果”
                if hits and hits[0].get('_score', 0) > 5:
                    has_rag_results = True
            except Exception:
                pass
            
            if has_rag_results:
                yield f"data: {json.dumps({'type': 'rag_start', 'content': '正在检索知识库...'}, ensure_ascii=False)}\n\n"
            
            rag_done_sent = False
            for response in bot.run(messages=messages):
                if not response:
                    continue
                
                # 第一次产出时，表示 RAG 检索完成
                if not rag_done_sent:
                    rag_done_sent = True
                    if has_rag_results:
                        yield f"data: {json.dumps({'type': 'rag_done', 'content': '知识库检索完成'}, ensure_ascii=False)}\n\n"
                
                if len(response) > prev_resp_len:
                    new_msgs = response[prev_resp_len:]
                    prev_resp_len = len(response)
                    
                    for msg in new_msgs:
                        if isinstance(msg, dict):
                            role = msg.get('role', '')
                            content = msg.get('content', '')
                            function_call = msg.get('function_call', None)
                            reasoning = msg.get('reasoning_content', '')
                        else:
                            role = getattr(msg, 'role', '')
                            content = getattr(msg, 'content', '')
                            function_call = getattr(msg, 'function_call', None)
                            reasoning = getattr(msg, 'reasoning_content', '')
                        
                        if reasoning:
                            event = {'type': 'thinking', 'content': str(reasoning)}
                            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                        
                        if function_call:
                            fc = function_call if isinstance(function_call, dict) else dict(function_call)
                            fc_key = f"{fc.get('name', '')}_{fc.get('arguments', '')[:50]}"
                            if fc_key not in seen_tool_calls:
                                seen_tool_calls.add(fc_key)
                                event = {
                                    'type': 'tool_call',
                                    'name': fc.get('name', ''),
                                    'arguments': fc.get('arguments', '')
                                }
                                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                        
                        if role == 'function':
                            name = msg.get('name', '') if isinstance(msg, dict) else getattr(msg, 'name', '')
                            result_key = f"{name}_{str(content)[:50]}"
                            if result_key not in seen_tool_results:
                                seen_tool_results.add(result_key)
                                event = {
                                    'type': 'tool_result',
                                    'name': name,
                                    'content': str(content)[:500] if content else ''
                                }
                                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                
                last_msg = response[-1] if response else None
                if last_msg:
                    if isinstance(last_msg, dict):
                        role = last_msg.get('role', '')
                        content = last_msg.get('content', '')
                        function_call = last_msg.get('function_call', None)
                    else:
                        role = getattr(last_msg, 'role', '')
                        content = getattr(last_msg, 'content', '')
                        function_call = getattr(last_msg, 'function_call', None)
                    
                    if role == 'assistant' and content and not function_call:
                        content_str = str(content)
                        current_idx = len(response) - 1
                        if current_idx != last_assistant_idx:
                            last_assistant_idx = current_idx
                            full_response_text = ""
                        
                        if len(content_str) > len(full_response_text):
                            delta = content_str[len(full_response_text):]
                            full_response_text = content_str
                            event = {
                                'type': 'text',
                                'delta': delta,
                                'full': full_response_text
                            }
                            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            # Save assistant reply to DB
            messages.append({'role': 'assistant', 'content': full_response_text})
            sessions[session_id] = messages[-20:]
            db_save_message(session_id, 'assistant', full_response_text)

            yield f"data: {json.dumps({'type': 'done', 'full': full_response_text}, ensure_ascii=False)}\n\n"
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# Mount static
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
