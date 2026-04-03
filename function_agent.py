"""
function_agent.py — FunctionAgent basic intro
Mirrors the official LlamaIndex AgentWorkflow example:
  https://developers.llamaindex.ai/python/examples/agent/agent_workflow_basic/

Uses Tavily for web search and the academy LLM instead of OpenAI.

Set env var:  TAVILY_API_KEY
"""

import asyncio
import os

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.agent.workflow import FunctionAgent
from tavily import AsyncTavilyClient


# ── LLM — OpenAI-compatible wrapper around the academy server ─────────────────
# The academy server exposes an OpenAI-compatible endpoint at /v1.
# OpenAILike handles async, streaming, and tool-calling automatically.

llm = OpenAILike(
    model="qwen3-80b",
    api_base="https://api.ukisai.academy/v1",
    api_key="dummy",               # server doesn't require a real key
    is_function_calling_model=True,
    is_chat_model=True,
    context_window=32000,
)


# ── Tool ──────────────────────────────────────────────────────────────────────

async def search_web(query: str) -> str:
    """Useful for using the web to answer questions."""
    client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return str(await client.search(query))


# ── Agent ─────────────────────────────────────────────────────────────────────

agent = FunctionAgent(
    tools=[search_web],
    llm=llm,
    system_prompt="You are a helpful assistant that can search the web for information.",
)


# ── Run ───────────────────────────────────────────────────────────────────────

async def main():
    # Basic run
    print("=== Basic run ===")
    response = await agent.run(user_msg="What is the weather in Belgrade right now?")
    print(str(response))

    # Maintaining state across turns
    print("\n=== Multi-turn with context ===")
    from llama_index.core.workflow import Context
    ctx = Context(agent)

    response = await agent.run(user_msg="My name is Jovan, nice to meet you!", ctx=ctx)
    print(str(response))

    response = await agent.run(user_msg="What is my name?", ctx=ctx)
    print(str(response))


asyncio.run(main())
