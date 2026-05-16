from langchain_core.runnables import RunnableLambda
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import OpenAI, ChatOpenAI

# model = OllamaLLM(  base_url="192.168.1.114:22026", model="gemma4", temperature=0)
model = ChatOpenAI(temperature=0)

documents = [
    Document(
        page_content=(
            "Classification is supervised learning. It uses labeled data "
            "to predict predefined categories such as spam or not spam."
        ),
        metadata={
            "source": "Module 6",
            "topic": "Classification",
        }
    ),
    Document(
        page_content=(
            "Clustering is unsupervised learning.  It groups similar data "
            "points without predefined labels."
        ),
        metadata={
            "source": "Module 7",
            "topic": "Clustering",
        }
    ),
    Document(
        page_content=(
            "Missing values can be handled by deletion, mean imputation, "
            "median imputation, or model-based imputation."
        ),
        metadata={
            "source": "Module 2",
            "topic": "Handling missing data",
        }
    ),
]


def retrieve_context(question: str) -> str:
    question_lower = question.lower()
    selected_docs = []
    formatted_docs = []

    for doc in documents:
        text = doc.page_content.lower()
        if any(word in text for word in question_lower.split()):
            selected_docs.append(doc)

    if not selected_docs:
        selected_docs = documents[:2]

    for doc in selected_docs:
        formatted_docs.append(f"source: {doc.metadata}\n Content: {doc.page_content}")

    return "\n\n".join(formatted_docs)


prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a careful data science instructor.
    Use only the provided context when answering.
    If the context is insufficient, say that the context is insufficient.
     
    """),
    ("human", """
    Question:
    {question}
    
    Context:
    {context}
    
    Answer:
    """)
])

rag_chain = (
        RunnableLambda(
            lambda question: {
                "question": question,
                "context": retrieve_context(question)
            }
        )
        | prompt
        | model
        | StrOutputParser()
)

answer = rag_chain.invoke(
    "Explain the difference between classification and clustering"
)

print(answer)
