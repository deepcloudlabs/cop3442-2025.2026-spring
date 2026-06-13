from dataclasses import dataclass, asdict
from datetime import datetime

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM


@dataclass
class PromptRunMetadata:
    prompt_id: str
    prompt_version: str
    model_name: str
    temparature: float
    timestamp: str

ROLE = "You are a senior AI Engineering instructor."
POLICY = "Use precise terminology and avoid unsupported claims."
FORMAT = "Return: definition, why it matters, and one implementation example."
MODEL_NAME = "gemma4"
TEMPERATURE = 0.0

meta = PromptRunMetadata(
    prompt_id="course_explainer",
    prompt_version="3.4.7",
    model_name=MODEL_NAME,
    temparature=TEMPERATURE,
    timestamp=datetime.utcnow().isoformat()
)

llm = OllamaLLM(model=MODEL_NAME, temperature=TEMPERATURE)
prompt = ChatPromptTemplate.from_messages([
    ("system", f"{ROLE}\n{POLICY}\n{FORMAT}"),
    ("human", "Explain {topic} in a concise technical paragraph.")
])

chain = prompt | llm | StrOutputParser

result = chain.invoke({
  "topic": "prompt versioning in LangChain applications"
})

run_record = {
    "metadata": asdict(meta),
    "input" : {
        "topic": "prompt versioning in LangChain applications"
    },
    "output": result
}

print(run_record)