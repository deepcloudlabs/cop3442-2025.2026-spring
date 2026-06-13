from langchain_core.prompts import ChatPromptTemplate

contract_prompt = ChatPromptTemplate.from_messages(
    [
        ("system","""
        you are a careful technical writer.
        Contract:
        -Use only the provided context.
        -Do not invent unsupported facts.
        -Return exactly three bullets.
        -Each bullet must be a sentence.
        """),
        ("human","Context:\n {context}\n\nTask:\n{task}")
    ]
)

result = contract_prompt.invoke({
    "context": "LCEL composes LangChain runnables using dataflow syntax.",
    "task": "Explain why LCEL is useful.",
})

print(result.to_string())