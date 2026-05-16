from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

#model = ChatOpenAI(temperature=0)
model = OllamaLLM(base_url="192.168.1.114:22026", model="gemma4", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an academic assistant for a university data science course.
      
Non-negotiable constraints:
- do not solve graded homework directly unless the instructor asks for a solution.
- When creating teaching materials, use original examples.
- Use practical examples: {use_examples}
- Explain assumptions clearly.
- Use Python and R examples when code is requested.
- Do not invent citations or unsupported facts.  
    """), (
        "human",
        """
        Create a short lecture explanation on {topic}.
        Difficulty level: {difficulty}
        Required format: {format}
        Code is requested: {use_code}
        """
    )]
)

chain = prompt | model | StrOutputParser()

result = chain.invoke({
    "topic": "ANOVA and post-hoc testing",
    "difficulty": "undergrad introductory level",
    "format": "five slide-style paragraphs",
    "use_examples": "Yes",
    "use_code": "Yes"
})

print(result)