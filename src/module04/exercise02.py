import json
from typing import Any

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama

def print_tool_trace(result: dict[str, Any]) -> None:
    """
    Print tool calls and tool results from the agent message sequence.
    """
    print("Tool Trace:")
    print("-" * 60)

    for message in result["messages"]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                print(f"Tool Call: {tool_call['name']}")
                print(f"Arguments: {tool_call['args']}")
                print()

        elif isinstance(message, ToolMessage):
            print(f"Tool Result from {message.name}:")
            print(message.content)
            print()

    print("-" * 60)


def print_final_answer(result: dict[str, Any]) -> None:
    """
    Print the last AI message with non-empty content.
    """
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and message.content:
            print(f"Agent: {message.content}\n")
            return

    print("Agent: No final answer was produced.\n")

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the internal knowledge base for company policies, procedures,
    and product information.

    Use this tool when the user asks about company-specific information,
    such as refund policy, shipping, warranty, or customer service hours.
    """
    knowledge_base = {
        "refund": (
            "Refund policy: Full refunds within 30 days. "
            "Partial refunds of 50% within 31-60 days. "
            "No refunds after 60 days."
        ),
        "shipping": (
            "Standard shipping: 5-7 days ($5). "
            "Express shipping: 2-3 days ($15). "
            "Overnight shipping: next day ($30). "
            "Free shipping for orders over $100."
        ),
        "warranty": (
            "All electronics have a 2-year warranty. "
            "Accessories have a 6-month warranty. "
            "Warranty claims can be filed at warranty.example.com."
        ),
        "hours": (
            "Customer service hours: Monday-Friday 9AM-8PM EST, "
            "Saturday 10AM-4PM EST, closed Sunday."
        ),
    }

    normalized_query = query.lower()

    for key, value in knowledge_base.items():
        if key in normalized_query:
            return value

    return "No relevant information found in the knowledge base."


@tool
def check_order_status(order_id: str) -> str:
    """
    Check the status of a customer order by order ID.

    Use this tool when the user provides an order number or asks about
    their order status.
    """
    print(f"Checking order status {order_id}")

    orders = {
        "ORD-12345": {
            "status": "Shipped",
            "carrier": "FedEx",
            "tracking": "FX789012",
            "eta": "2025-03-20",
        },
        "ORD-67890": {
            "status": "Processing",
            "carrier": None,
            "tracking": None,
            "eta": "2025-03-25",
        },
        "ORD-11111": {
            "status": "Delivered",
            "carrier": "UPS",
            "tracking": "UP456789",
            "eta": None,
        },
    }

    normalized_order_id = order_id.strip().upper()
    order = orders.get(normalized_order_id)

    if order is None:
        return f"Order {normalized_order_id} not found. Please verify the order ID."

    return json.dumps(order, indent=2)


@tool
def calculate_refund(amount: float, days_since_purchase: int) -> str:
    """
    Calculate the refund amount based on the purchase price and the number
    of days since purchase.

    Use this tool when the customer asks how much refund they would receive.
    """
    print(f"Calculating refund amount {amount} and days {days_since_purchase}")
    if days_since_purchase <= 30:
        refund = amount
        policy = "full refund within 30 days"
    elif days_since_purchase <= 60:
        refund = amount * 0.5
        policy = "50% partial refund within 31-60 days"
    else:
        refund = 0.0
        policy = "no refund after 60 days"

    return (
        f"Refund amount: ${refund:.2f}\n"
        f"Policy applied: {policy}\n"
        f"Original purchase amount: ${amount:.2f}\n"
        f"Days since purchase: {days_since_purchase}"
    )


tools = [
    search_knowledge_base,
    check_order_status,
    calculate_refund
]

SYSTEM_PROMPT = """
You are a customer service agent for TechMart, an electronics retailer.

Guidelines:
1. Greet the customer warmly when appropriate.
2. Use search_knowledge_base for policy questions.
3. Use check_order_status when the customer mentions an order ID.
4. Use calculate_refund when the customer asks about refund amounts.
5. If a tool returns no results, acknowledge this and ask the customer to clarify.
6. Never make up information. Use only information returned by tools.
7. After resolving the issue, ask if there is anything else you can help with.
8. Be empathetic, concise, and solution-oriented.
"""

llm = ChatOllama(model="gemma4", temperature=0.0)

agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)

conversation = []

user_message_1 = "Hi! Can you check on my order ORD-12345?"
conversation.append(HumanMessage(content=user_message_1))

result_1 = agent.invoke({"messages": conversation},
             config={"recursion_limit": 10}
             )

print_final_answer(result_1)

conversation = result_1["messages"]

user_message_2 = (
    "I want to return it. I paid $299.99 and bought it 25 days ago."
    "What refund would i get?"
)

conversation.append(HumanMessage(content=user_message_2))

result_2 = agent.invoke({"messages": conversation},
             config={"recursion_limit": 10}
             )

print_final_answer(result_2)

print_tool_trace(result_2)

