from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {"question": "What is RAG?", "answer": "RAG injects retrieved evidence before generation."},
    {"question": "What is parsing?", "answer": "Parsing converts raw model text into structured data."},
]

example_prompt = PromptTemplate.from_template(
    "Q: {question}\nA: {answer}"
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Answer in one precise sentence.",
    suffix="Q: {input}\nA:",
    input_variables=["input"],
)

print(prompt.invoke({"input": "What is LCEL?"}).text)