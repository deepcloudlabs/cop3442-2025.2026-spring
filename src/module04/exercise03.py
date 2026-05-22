import os
from typing import Any, ClassVar, Optional

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool, ToolException, tool


llm = ChatOllama(model="gemma4", temperature=0.0, )



# Simple tool with @tool decorator
@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a city.

    Use this tool when the user asks about weather or needs weather
    information for travel planning.

    Input:
    - city: city name, such as London, New York, or Tokyo.
    """
    weather_data = {
        "london": {
            "temp": "12°C",
            "conditions": "Cloudy",
            "humidity": "78%",
        },
        "new york": {
            "temp": "22°C",
            "conditions": "Sunny",
            "humidity": "45%",
        },
        "tokyo": {
            "temp": "18°C",
            "conditions": "Partly Cloudy",
            "humidity": "62%",
        },
    }

    normalized_city = city.strip().lower()
    data = weather_data.get(normalized_city)

    if data is None:
        return f"Weather data is not available for {city}."

    return (
        f"Weather in {city}: "
        f"{data['temp']}, "
        f"{data['conditions']}, "
        f"humidity: {data['humidity']}."
    )


# Structured Tool with Pydantic Schema

class FlightSearchInput(BaseModel):
    origin: str = Field(
        description="Origin airport code, for example JFK or LAX."
    )
    destination: str = Field(
        description="Destination airport code, for example LHR or NRT."
    )
    date: str = Field(
        description="Travel date in YYYY-MM-DD format."
    )
    passengers: int = Field(
        default=1,
        ge=1,
        le=9,
        description="Number of passengers. Must be between 1 and 9.",
    )


def search_flights_impl(
    origin: str,
    destination: str,
    date: str,
    passengers: int = 1,
) -> str:
    """
    Search for available flights.
    """
    flights = [
        {
            "airline": "United",
            "flight": "UA100",
            "depart": "08:00",
            "arrive": "20:00",
            "price": 850,
        },
        {
            "airline": "British Airways",
            "flight": "BA178",
            "depart": "19:30",
            "arrive": "07:30+1",
            "price": 1200,
        },
    ]

    results = []

    for flight in flights:
        total_price = flight["price"] * passengers

        results.append(
            f"{flight['airline']} {flight['flight']}: "
            f"{flight['depart']}-{flight['arrive']}, "
            f"${total_price} total for {passengers} passenger(s)"
        )

    return (
        f"Flights from {origin.upper()} to {destination.upper()} "
        f"on {date}:\n"
        + "\n".join(results)
    )


flight_search = StructuredTool.from_function(
    func=search_flights_impl,
    name="search_flights",
    description=(
        "Search for available flights between airports on a specific date. "
        "Use this tool when the user wants to find flights or plan air travel. "
        "Requires origin airport code, destination airport code, travel date, "
        "and optionally the number of passengers."
    ),
    args_schema=FlightSearchInput,
    handle_tool_error=True,
)


# Custom Tool Class inherited from BaseTool

class CurrencyConverterInput(BaseModel):
    amount: float = Field(
        gt=0,
        description="Amount to convert. Must be greater than zero.",
    )
    from_currency: str = Field(
        description="Source currency code, for example USD, EUR, GBP, or JPY.",
    )
    to_currency: str = Field(
        description="Target currency code, for example USD, EUR, GBP, or JPY.",
    )


class CurrencyConverterTool(BaseTool):
    name: str = "currency_converter"
    description: str = (
        "Convert an amount from one currency to another using predefined "
        "exchange rates. Use this tool when the user asks about currency "
        "conversion or travel budgeting."
    )
    args_schema: type[BaseModel] = CurrencyConverterInput

    rates: ClassVar[dict[tuple[str, str], float]] = {
        ("USD", "EUR"): 0.92,
        ("EUR", "USD"): 1.09,
        ("USD", "GBP"): 0.79,
        ("GBP", "USD"): 1.27,
        ("USD", "JPY"): 149.50,
        ("JPY", "USD"): 0.0067,
    }

    def _run(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        source = from_currency.strip().upper()
        target = to_currency.strip().upper()

        if source == target:
            return f"{amount:.2f} {source} = {amount:.2f} {target}"

        rate = self.rates.get((source, target))

        if rate is None:
            return f"Exchange rate for {source} to {target} is not available."

        converted = amount * rate

        return (
            f"{amount:.2f} {source} = {converted:.2f} {target} "
            f"(exchange rate: {rate})"
        )


currency_converter = CurrencyConverterTool()


# Tool with Custom Error Handling

class HotelBookingInput(BaseModel):
    city: str = Field(description="Hotel city, for example London or Tokyo.")
    check_in: str = Field(description="Check-in date in YYYY-MM-DD format.")
    check_out: str = Field(description="Check-out date in YYYY-MM-DD format.")
    guests: int = Field(
        default=1,
        ge=1,
        description="Number of guests.",
    )


def custom_tool_error_handler(error: ToolException) -> str:
    return (
        f"Hotel booking tool error: {error}. "
        f"Please ask the user to revise the booking details."
    )


def book_hotel_impl(
    city: str,
    check_in: str,
    check_out: str,
    guests: int = 1,
) -> str:
    """
    Book a hotel in a city for the specified dates.
    """
    if guests > 4:
        raise ToolException(
            "Maximum 4 guests per room. Split the request into multiple bookings."
        )

    confirmation_code = f"HTL-{abs(hash((city, check_in, check_out))) % 10000:04d}"

    return (
        f"Hotel booked in {city}: "
        f"{check_in} to {check_out}, "
        f"{guests} guest(s). "
        f"Confirmation: {confirmation_code}"
    )


book_hotel = StructuredTool.from_function(
    func=book_hotel_impl,
    name="book_hotel",
    description=(
        "Book a hotel in a city for specified check-in and check-out dates. "
        "Use this tool when the user explicitly asks to book a hotel. "
        "Maximum 4 guests per room."
    ),
    args_schema=HotelBookingInput,
    handle_tool_error=custom_tool_error_handler,
)


# Assemble Agent

all_tools = [
    get_weather,
    flight_search,
    currency_converter,
    book_hotel,
]

SYSTEM_PROMPT = """
You are a travel planning assistant.

You can help users by:
1. Checking destination weather.
2. Searching for flights.
3. Converting currencies for budgeting.
4. Booking hotels.

Tool-use rules:
1. Use get_weather when the user asks about weather.
2. Use search_flights when the user asks about flights.
3. Use currency_converter when the user asks about currency conversion.
4. Use book_hotel only when the user explicitly asks to book a hotel.
5. If a tool returns unavailable information, explain the limitation clearly.
6. Do not invent live travel, weather, flight, hotel, or currency data.
7. Be concise, practical, and helpful.
"""

travel_agent = create_agent(
    model=llm,
    tools=all_tools,
    system_prompt=SYSTEM_PROMPT,
)


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def print_final_answer(result: dict[str, Any]) -> None:
    """
    Print the final AI response from the agent result.
    """
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and message.content:
            print("\nAgent:")
            print(message.content)
            return

    print("\nAgent:")
    print("No final answer was produced.")


def print_tool_trace(result: dict[str, Any]) -> None:
    """
    Print tool calls and tool observations for debugging.
    """
    print("\nTool Trace:")
    print("=" * 80)

    for message in result["messages"]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                print(f"Tool call: {tool_call['name']}")
                print(f"Arguments: {tool_call['args']}")
                print("-" * 80)

        elif isinstance(message, ToolMessage):
            print(f"Tool result from {message.name}:")
            print(message.content)
            print("-" * 80)


# ============================================================
# 7. RUN AGENT
# ============================================================

user_request = (
    "I want to fly from JFK to London on 2025-04-15. "
    "What's the weather like there? "
    "Also, how much is $500 in British Pounds?"
)

result = travel_agent.invoke(
    {
        "messages": [
            HumanMessage(content=user_request)
        ]
    },
    config={
        "recursion_limit": 8
    },
)

print_final_answer(result)
print_tool_trace(result)