# Conversation history with MessagesPlaceholder
# pip install -U langchain langchain_ollama

import os

from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

model = OllamaLLM(model=os.getenv("OLLAMA_MODEL", "gemma4"), temperature=0.0)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a data science teaching assistant. Maintain continuity with the conversation."
    ),
    MessagesPlaceholder(variable_name="history"),
    (
        "human",
        "{question}"
    )
])

chain = prompt | model | StrOutputParser()

history = [
    HumanMessage(content="We are studying data cleaning."),
    AIMessage(content="Good. Data cleaning includes missing values, duplicates, outliers, and transformations."),
    HumanMessage(content="We also discussed median imputation."),
    AIMessage(content="Median imputation is robust when distributions are skewed or contain outliers.")
]

result = chain.invoke({
    "history": history,
    "question": "Now explain how outlier handling relates to median imputation."
})

print(result)