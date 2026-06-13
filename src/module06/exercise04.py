from langchain_core.prompts import ChatPromptTemplate

ROLE = "You are a senior AI Engineering instructor."
POLICY = "Use precise terminology and avoid unsupported claims."
FORMAT = "Return: definition, why it matters, and one implementation example."
prompt = ChatPromptTemplate.from_messages([
    ("system", f"{ROLE}\n{POLICY}\n{FORMAT}"),
    ("human", "Topic: {topic}\nAudience: {audience}")
])

response = prompt.invoke({
    "topic": "prompt modularity",
    "audience": "graduate AI engineering students"
})

print(response.to_string())
