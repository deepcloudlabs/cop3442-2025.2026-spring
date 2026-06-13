from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template("""
Course: {course}
Policy: {policy}

Topic: {topic}
Task: Explain the topic for {audience}.
""")

prompt = template.partial(
    course="Prompt Engineering",
    policy="Use technical language and avoid unsupported claims."
)

print(prompt.invoke({
    "topic": "partial variables in LangChain",
    "audience": "AI engineering students"
}).text)