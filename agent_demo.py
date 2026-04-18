"""
agent_demo.py — OpenClaw-style agent with 3 tools
Built on function_agent.py patterns (async FunctionAgent + OpenAILike)

The agent can:
  1. search_web    →  Tavily  (live info from the internet)
  2. create_note   →  create a new .md file and register it in SKILLS.md
  3. edit_file     →  edit an existing file by replacing a section of text

SKILLS.md is loaded at startup and injected into the system prompt so the
agent always knows what knowledge files exist and what they contain.

Pre-requisites:
    pip install llama-index-llms-openai-like tavily-python
    export TAVILY_API_KEY=your_key
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.agent.workflow import FunctionAgent
from tavily import AsyncTavilyClient

load_dotenv()


SKILLS_FILE = Path("SKILLS.md")
UKISAI_KNOWLEDGE_FILE = Path("skills/ukisai.md")


# ── LLM ───────────────────────────────────────────────────────────────────────

llm = OpenAILike(
    model="qwen3-80b",
    api_base="https://api.ukisai.academy/v1",
    api_key="dummy",
    is_function_calling_model=True,
    is_chat_model=True,
    context_window=32000,
)


# ── Tool 1: Web search (copied from function_agent.py) ───────────────────────

async def search_web(query: str) -> str:
    """Useful for using the web to answer questions."""
    client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return str(await client.search(query))


# ── Tool 2: Create note ───────────────────────────────────────────────────────
# WRITE THIS IN CLASS ↓

def create_note(filename: str, description: str, content: str) -> str:
    """
    Create a new markdown file and register it in SKILLS.md so the agent
    remembers what the file is for in future conversations.
    filename:    e.g. 'notes/python_tips.md'
    description: one-line summary shown in SKILLS.md index
    content:     full markdown content to write into the file
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    # Register in SKILLS.md
    entry = f"- **[{path.name}]({filename})** — {description}\n"
    skills_text = SKILLS_FILE.read_text(encoding="utf-8")
    updated = skills_text.replace(
        "<!-- entries added automatically by create_note tool -->",
        f"<!-- entries added automatically by create_note tool -->\n{entry}",
    )
    SKILLS_FILE.write_text(updated, encoding="utf-8")

    return f"Created {path.resolve()} and registered in SKILLS.md"


# ── Tool 3: Edit file ─────────────────────────────────────────────────────────
# WRITE THIS IN CLASS ↓

def edit_file(filename: str, old_text: str, new_text: str) -> str:
    """
    Edit an existing file by replacing old_text with new_text.
    The file must already exist.
    filename: path to the file
    old_text: exact text to find and replace (must be unique in the file)
    new_text: text to replace it with
    """
    path = Path(filename)
    if not path.exists():
        return f"Error: {filename} does not exist. Use create_note to create it first."

    original = path.read_text(encoding="utf-8")
    if old_text not in original:
        return f"Error: could not find the text to replace in {filename}."

    path.write_text(original.replace(old_text, new_text, 1), encoding="utf-8")
    return f"Edited {filename} successfully."


# ── Load SKILLS.md into system prompt ─────────────────────────────────────────

def load_skills() -> str:
    if SKILLS_FILE.exists():
        return SKILLS_FILE.read_text(encoding="utf-8")
    return "(no SKILLS.md found)"


def load_ukisai_knowledge() -> str:
    if UKISAI_KNOWLEDGE_FILE.exists():
        return UKISAI_KNOWLEDGE_FILE.read_text(encoding="utf-8")
    return "(no skills/ukisai.md found)"


UKISAI_SYSTEM_PROMPT = """You are an AI assistant for UkisAI.

Your role is to clearly explain what UkisAI does, answer questions about UkisAI's products, services, and capabilities, and help users understand how UkisAI can help them.

You must:
- Only answer questions related to UkisAI.
- Use the provided context and conversation history to give accurate, relevant answers.
- Be concise, clear, and helpful.
- Ask clarifying questions if the user's request is unclear.

You must NOT:
- Answer questions unrelated to UkisAI.
- Provide general knowledge, opinions, or information outside of UkisAI.
- Make up information if you are unsure.

If a question is not related to UkisAI, politely refuse and guide the user back to UkisAI-related topics."""


# ── Agent ─────────────────────────────────────────────────────────────────────

agent = FunctionAgent(
    tools=[search_web, create_note, edit_file],
    llm=llm,
    system_prompt=(
        UKISAI_SYSTEM_PROMPT
        + "\n\n## UkisAI knowledge base\n\n"
        + load_ukisai_knowledge()
        + "\n\n## Skills index (SKILLS.md)\n\n"
        + load_skills()
        + "\n\nYou have tools: search_web, create_note, and edit_file. "
        "Use search_web only for factual questions strictly about UkisAI when the knowledge base above "
        "is insufficient; do not use web search for unrelated topics. "
        "Use create_note and edit_file only for UkisAI-related notes or documentation."
    ),
)


# ── Chat loop ─────────────────────────────────────────────────────────────────

from llama_index.core.workflow import Context


async def chat():
    ctx = Context(agent)
    print("UkisAI chatbot — type 'exit' or 'quit' to leave.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        response = await agent.run(user_msg=question, ctx=ctx)
        print(f"\nBot: {response}\n")


if __name__ == "__main__":
    asyncio.run(chat())
