from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

repair_prompt = PromptTemplate.from_template("""
You are a JSON repair assistant.
Return only valid JSON.
Do not add explanations.

Schema requirement:
{schema}

Invalid output:
{bad_output}

Parser error:
{error}
""")

llm = OllamaLLM(model="gemma4", temperature=0.0)
repair_chain = repair_prompt | llm | StrOutputParser()

print(repair_chain.invoke({
    "schema": '{"grade": integer, "feedback": string}',
    "bad_output": '{grade: excellent, feedback: Good}',
    "error": "grade must be an integer; keys must be quoted"
}))
