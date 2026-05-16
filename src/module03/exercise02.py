from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import OllamaLLM

model = OllamaLLM(base_url="192.168.1.114:12026", model="gemma4", temperature=0)

messages = [
    SystemMessage(
        content=(
            "You are a data science professor"
            "Use academic language, define key terms, and provide explanation with examples"
        )
    ),
    HumanMessage(
        content=(
            "Explain why missing value imputation is necessary before building"
            "a machine learning model."
        )
    )
]

response = model.invoke(messages)

print(response)
