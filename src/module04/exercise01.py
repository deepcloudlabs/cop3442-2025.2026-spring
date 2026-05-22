from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma4", temperature=0.0)

def simple_conversation_loop():
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant.You can answer questions and help with tasks"
         "If you need more information, ask the user."
         "If the user says 'bye', respond with 'Goodbye!' and nothing else."
         ),
        MessagesPlaceholder(variable_name="history"),
        ("human","{input}")
    ])

    chain = prompt | llm | StrOutputParser()

    history = []

    user_inputs = [
        "Hi! I am planning a trip to Belgium.",
        "What is the best time to visit?",
        "Any must-see places?",
        "bye"
    ]

    for user_input in user_inputs:
        # perception: receive user input + history
        # reasoning + action: LLM generates the response
        response = chain.invoke({"input": user_input, "history": history})

        # update state -> memory
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response))

        print(f"User: {user_input}")
        print(f"Agent: {response}")

        # termination condition!
        if "goodbye" in response.lower():
            break
            
simple_conversation_loop()