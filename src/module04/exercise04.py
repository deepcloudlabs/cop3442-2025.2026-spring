from langchain.agents import create_agent
from langchain_core.runnables import configurable
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

model = ChatOllama(model="gemma4", temperature=0.0)

agent = create_agent(
    model=model,
    tools=[],
    system_prompt=(
        "You are a concise AI engineering tutor. "
        "Remember facts stated by the user within the current thread."
    ),
    #checkpointer=InMemorySaver()
)

config = {
    "configurable": {
        "thread_id": "ain2002-session-001"
    }
}

agent.invoke({
    "messages": [
        {"role": "user", "content": "My course is AIN-2002."}
    ]
}, config
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Which course code did i mention?"}
    ]
}, config
)

print(result["messages"][-1].content)

