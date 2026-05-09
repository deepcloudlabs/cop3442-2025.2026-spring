from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(
    base_url="192.168.1.111:12026",
    model = "gemma4",
    temperature=0.0
)

prompt = PromptTemplate.from_template("""
Your are a {lecture_style} lecturer.

Grade the student's quiz answer between {min_grade} and {max_grade}

Question:
{question}

Reference answer:
{reference_answer}

Student answer:
{student_answer}

Return only:
Grade: <score>/{max_grade}
Feedback: <short feedback>
""")

chain = prompt | llm | StrOutputParser()


response = chain.invoke({
    "lecture_style": "strict",
    "question": "Explain what self-attention does in a Transformer model",
    "reference_answer": r"Self-attention is a mechanism that allows a neural network to weigh the importance of different words or parts of an input sequence relative to a specific word within that same sequence, thereby capturing deep contextual relationships. Instead of processing words sequentially, it simultaneously calculates how much every element should pay attention to every other element. This process is mathematically achieved by generating three vectors for each input: a Query (Q), a Key (K), and a Value (V). The similarity between the Query of the current word and the Keys of all other words determines the attention weights, which are then used to create a weighted sum of the Values. This weighted sum produces a new representation for the word that is rich with context, enabling the model to understand that the meaning of a word (like 'it') depends on all the other words in the sentence, regardless of how far apart they are.",
     "student_answer": "self-attention is a mechanism",
     "min_grade": 0,
     "max_grade": 10
})

print(response)