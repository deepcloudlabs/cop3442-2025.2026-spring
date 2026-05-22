"""
Supervisor agent that calls specialized subagents as tools.
"""
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

model = ChatOllama(model="gemma4", temperature=0.0)

research_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="You are a research assistant. Return concise conceptual findings."
)

coding_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="You are a Python coding assistant. Return runnable Python examples."
)


@tool
def call_research_agent(query: str) -> str:
    """Call the research subagent for conceptual explanation tasks."""
    response = research_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return response["messages"][-1].content


@tool
def call_coding_agent(query: str) -> str:
    """Call the coding subagent for Python implementation tasks."""
    response = coding_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return response["messages"][-1].content


supervisor_agent = create_agent(
    model=model,
    tools=[call_research_agent, call_coding_agent],
    system_prompt=(
        "You are a supervisor agent. Use the research subagent for concepts, "
        "the coding subagent for implementation, and synthesize a final answer."
    )
)

result = supervisor_agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Explain conversational memory conceptually and provide a Python implementation idea."
        }
    ]
})

print(result["messages"][-1].content)
