import os
import re
from typing import TypedDict

import ollama
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from tools import (
    calculate_math,
    get_current_date,
    search_knowledge_base,
)


load_dotenv()

MODEL_NAME = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:latest"
)


class AgentState(TypedDict, total=False):
    question: str
    route: str
    tool_result: str
    final_answer: str


def is_relative_date_query(
    question: str
) -> bool:
    """
    Detects requests for today's, yesterday's,
    tomorrow's, or the current date/time.

    Birth-date and historical-year questions are
    not current-date requests.
    """

    q = question.lower()

    knowledge_phrases = [
        "date of birth",
        "birth date",
        "birthday",
        "born",
        "happened in",
    ]

    if any(
        phrase in q
        for phrase in knowledge_phrases
    ):
        return False

    date_phrases = [
        "today",
        "today's date",
        "yesterday",
        "yeterday",
        "tomorrow",
        "current date",
        "current time",
        "what time is it",
    ]

    return any(
        phrase in q
        for phrase in date_phrases
    )


def is_math_query(question: str) -> bool:
    """
    Detects actual calculations.

    A number by itself, such as 1947, is not
    automatically treated as math.
    """

    q = question.lower().strip()

    math_words = [
        "calculate",
        "solve",
        "plus",
        "minus",
        "multiply",
        "multiplied by",
        "divide",
        "divided by",
        "percentage",
        "times",
        "modulo",
    ]

    if any(
        word in q
        for word in math_words
    ):
        return True

    contains_number = bool(
        re.search(r"\d", q)
    )

    contains_operator = bool(
        re.search(
            r"[+\-*/%]",
            q
        )
    )

    return (
        contains_number
        and contains_operator
    )


def router_node(
    state: AgentState
) -> AgentState:
    """
    Selects the tool according to the question.
    """

    question = state["question"]

    if is_relative_date_query(question):
        route = "date"

    elif is_math_query(question):
        route = "calculator"

    else:
        route = "knowledge_base"

    return {
        **state,
        "route": route
    }


def date_node(
    state: AgentState
) -> AgentState:
    """
    Executes the Date Tool.
    """

    result = get_current_date(
        state["question"]
    )

    return {
        **state,
        "tool_result": result
    }


def calculator_node(
    state: AgentState
) -> AgentState:
    """
    Executes the Calculator Tool.
    """

    result = calculate_math(
        state["question"]
    )

    return {
        **state,
        "tool_result": result
    }


def knowledge_base_node(
    state: AgentState
) -> AgentState:
    """
    Executes the Knowledge Base Tool.
    """

    result = search_knowledge_base(
        state["question"]
    )

    return {
        **state,
        "tool_result": result
    }


def final_answer_node(
    state: AgentState
) -> AgentState:
    """
    Uses Qwen3 to produce a clear final response.
    """

    route = state["route"]
    question = state["question"]

    tool_result = state.get(
        "tool_result",
        ""
    )

    if route == "date":
        prompt = f"""
User question:
{question}

Date Tool result:
{tool_result}

Return the Date Tool result clearly and briefly.
Do not change the date.
"""

    elif route == "calculator":
        prompt = f"""
User question:
{question}

Calculator Tool result:
{tool_result}

Give the mathematical result clearly.
Do not perform a different calculation.
"""

    else:
        prompt = f"""
You are a helpful agentic AI chatbot.

First use the local knowledge-base context below.

If the local context contains the answer, answer from
that context.

If the local context explicitly says that sufficient
information was not found, answer using your general
knowledge. If you are uncertain, state that clearly
instead of inventing information.

User question:
{question}

Local knowledge-base context:
{tool_result}

Final answer:
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response[
        "message"
    ]["content"].strip()

    return {
        **state,
        "final_answer": answer
    }


def route_condition(
    state: AgentState
) -> str:
    """
    Returns the route selected by the router.
    """

    return state["route"]


def build_graph():
    """
    Builds and compiles the LangGraph workflow.
    """

    graph = StateGraph(AgentState)

    graph.add_node(
        "router",
        router_node
    )

    graph.add_node(
        "date",
        date_node
    )

    graph.add_node(
        "calculator",
        calculator_node
    )

    graph.add_node(
        "knowledge_base",
        knowledge_base_node
    )

    graph.add_node(
        "final_answer",
        final_answer_node
    )

    graph.add_edge(
        START,
        "router"
    )

    graph.add_conditional_edges(
        "router",
        route_condition,
        {
            "date": "date",
            "calculator": "calculator",
            "knowledge_base": "knowledge_base",
        }
    )

    graph.add_edge(
        "date",
        "final_answer"
    )

    graph.add_edge(
        "calculator",
        "final_answer"
    )

    graph.add_edge(
        "knowledge_base",
        "final_answer"
    )

    graph.add_edge(
        "final_answer",
        END
    )

    return graph.compile()


def terminal_chatbot() -> None:
    """
    Runs the terminal chatbot.
    """

    app = build_graph()

    print("===== Agentic AI Terminal Chatbot =====")
    print(f"Model: {MODEL_NAME}")

    print("This chatbot can:")
    print("1. Search the knowledge base")
    print("2. Calculate math")
    print("3. Check the relative/current date")
    print("4. Generate a final answer")
    print("Type 'exit' to quit.")

    while True:

        question = input("\nYou: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not question:
            continue

        try:

            result = app.invoke(
                {
                    "question": question
                }
            )

            route_names = {
                "date": "Date Tool",
                "calculator": "Calculator Tool",
                "knowledge_base": "Knowledge Base Tool",
            }

            selected_route = result["route"]

            print("\n================================")
            print("[Agent Decision]")
            print(
                f"Selected Tool   : {route_names[selected_route]}"
            )
            print("================================")

            print("\n[Final Answer]")
            print(result["final_answer"])

        except Exception as error:

            print("\nError:")
            print(error)

            print("\nPlease check:")
            print("1. Ollama is running")
            print("2. qwen3:latest exists in ollama list")
            print("3. nomic-embed-text:latest exists in ollama list")
            print("4. The database was built using build_db.py")
            print("5. Required Python packages are installed")
if __name__ == "__main__":
    terminal_chatbot()
