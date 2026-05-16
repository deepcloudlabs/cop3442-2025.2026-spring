from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# model = ChatOpenAI(temperature=0)

model = OllamaLLM(model="gemma4", temperature=0.0)

examples = [
    {
        "input": "Classification",
        "output": (
            "Classification is a supervised learning task where the model "
            "learns from labeled examples and predicts predefined categories."
        )
    },
    {
        "input": "Clustering",
        "output": (
            "Clustering is an unsupervised learning task where the model "
            "groups similar observations without predefined labels."
        )
    },
    {
        "input": "PCA",
        "output": (
            "Principal Component Analysis is a dimensionality reduction "
            "method that transforms correlated features into orthogonal "
            "components while preserving maximum variance."
        )
    }
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "Explain this concept: {input}"),
    ("ai", "{output}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

final_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a data science instructor. Explain concepts clearly and concisely."
    ),
    few_shot_prompt,
    (
        "human",
        "Explain this concept: {concept}"
    )
])

chain = final_prompt | model | StrOutputParser()

result = chain.invoke({
    "concept": "Silhouette score"
})

print(result)
