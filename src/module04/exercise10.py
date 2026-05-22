"""
Simple LangGraph workflow with routing.
"""

import os
from typing import Literal

from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage


class WorkflowState(TypedDict):
    user_request: str
    category: Literal["teaching", "coding", "other"]
    answer: str


model = ChatOllama(model="gemma4", temperature=0.0)


def classify_request(state: WorkflowState) -> dict:
    response = model.invoke([
        HumanMessage(
            content=(
                "Classify the request as exactly one of: teaching, coding, other.\n"
                f"Request: {state['user_request']}"
            )
        )
    ])
    text = response.content.lower()
    if "coding" in text:
        return {"category": "coding"}
    if "teaching" in text:
        return {"category": "teaching"}
    return {"category": "other"}


def teaching_node(state: WorkflowState) -> dict:
    response = model.invoke([
        HumanMessage(content=f"Write a teaching explanation for: {state['user_request']}")
    ])
    return {"answer": response.content}


def coding_node(state: WorkflowState) -> dict:
    response = model.invoke([
        HumanMessage(content=f"Write Python-oriented implementation guidance for: {state['user_request']}")
    ])
    return {"answer": response.content}


def fallback_node(state: WorkflowState) -> dict:
    return {"answer": "The request is outside the supported workflow."}


def route(state: WorkflowState) -> str:
    return state["category"]


builder = StateGraph(WorkflowState)
builder.add_node("classify", classify_request)
builder.add_node("teaching", teaching_node)
builder.add_node("coding", coding_node)
builder.add_node("other", fallback_node)

builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify",
    route,
    {"teaching": "teaching", "coding": "coding", "other": "other"}
)
builder.add_edge("teaching", END)
builder.add_edge("coding", END)
builder.add_edge("other", END)

graph = builder.compile()

result = graph.invoke({
    "user_request": "Explain LangGraph for undergraduate AI engineering students."
})

print(result["category"])
print(result["answer"])
