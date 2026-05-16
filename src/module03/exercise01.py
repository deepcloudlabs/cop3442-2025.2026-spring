# Prompt assembly with runtime variables
# pip install -U langchain langchain_ollama

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = OllamaLLM(base_url="192.168.1.114:22026", model="gemma4", temperature=0)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
      You are a senior data science instructor.
      
      Teaching constraints:
      - target audience: {audience}
      - course level: {course_level}
      - Use practical examples: {use_examples}
      - avoid unnecessary mathematical complexity unless requested.
    """),
    ("human", """
    Topic: {topic}
    
    Prepare a clear explanation that can be converted into presentation slides.
    """)
])

chain = prompt_template | model | StrOutputParser()

result = chain.invoke({
    "audience": "second-year AI Engineering Students",
    "course_level": "introduction",
    "use_examples": "Yes",
    #"topic": "K-Means Clustering",
    "topic": "Silhouette score",
})

print(result)
