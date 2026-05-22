"""
Simple reliability test harness for an LLM chain.
The test checks whether required terms appear in each answer.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="gemma4", temperature=0.0)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a data science instructor. Always include definition, example, and limitation."
    ),
    (
        "human",
        "Explain {topic}."
    )
])

chain = prompt | model | StrOutputParser()

topics = [
    "conversational agency",
    "tool calling",
    "short-term memory",
    "workflow routing"
]

required_terms = ["definition", "example", "limitation"]

for topic in topics:
    output = chain.invoke({"topic": topic})
    output_lower = output.lower()
    print("=" * 80)
    print("TOPIC:", topic)
    print(output)
    print("\nEVALUATION")
    for term in required_terms:
        print(f"Contains {term!r}:", term in output_lower)
