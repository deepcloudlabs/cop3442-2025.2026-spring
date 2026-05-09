from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(
    base_url="192.168.1.111:12026",
    model = "gemma4",
    temperature=0.2
)

prompt = "Explain self-attention in one paragraph"

response = llm.invoke([
    HumanMessage(content=prompt)
])

print(response.content)