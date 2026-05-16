# Parameterized task specification
# pip install -U langchain langchain-ollama

from langchain_ollama import OllamaLLM


model = OllamaLLM( base_url="192.168.1.114:12026", model="gemma4", temperature=0)

template = PromptTemplate.from_template("""
You are preparing teaching material for the course {course_code}: {course_title}.

Task:
{task}

Topic:
{topic}

Audience:
{audience}

Output Requirements:
- Define the concept.
- Explain why it matters
- Provide one realistic example.
- Provide one {programming_language}-oriented implementation.
- Keep the explanation suitable for presentation slides.
""")

chain = template | model | StrOutputParser()

result = chain.invoke({
    "course_code": "AIN-2002",
    "course_title": "Introduction to Data Science",
    "task": "Create presentation-ready teaching notes",
    "topic": "Data cleaning and missing value imputation",
    "audience": "undergraduate AI engineering students",
    "programming_language": "javascript",
})

print(result)