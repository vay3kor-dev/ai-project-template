import ast
import operator
import re
from datetime import datetime, timedelta
from pathlib import Path

import chromadb
import ollama


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "agentic_kb"
EMBEDDING_MODEL = "nomic-embed-text"


# ============================================================
# DATE TOOL
# ============================================================

def get_current_date(question: str = "") -> str:
    """
    Returns today's, yesterday's, or tomorrow's date.
    """

    query = question.lower().strip()

    if "yesterday" in query or "yeterday" in query:
        requested_date = datetime.now() - timedelta(days=1)

        return (
            "Yesterday's date was "
            f"{requested_date.strftime('%d %B %Y')}."
        )

    if "tomorrow" in query:
        requested_date = datetime.now() + timedelta(days=1)

        return (
            "Tomorrow's date is "
            f"{requested_date.strftime('%d %B %Y')}."
        )

    requested_date = datetime.now()

    return (
        "Today's date is "
        f"{requested_date.strftime('%d %B %Y')}."
    )


# ============================================================
# CALCULATOR TOOL
# ============================================================

ALLOWED_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate_math_node(node):
    """
    Safely evaluates arithmetic syntax.
    """

    if isinstance(node, ast.Expression):
        return evaluate_math_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value

        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)

        if operator_type not in ALLOWED_BINARY_OPERATORS:
            raise ValueError(
                "This mathematical operator is not supported."
            )

        left_value = evaluate_math_node(node.left)
        right_value = evaluate_math_node(node.right)

        operation = ALLOWED_BINARY_OPERATORS[operator_type]

        return operation(left_value, right_value)

    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)

        if operator_type not in ALLOWED_UNARY_OPERATORS:
            raise ValueError(
                "This unary operator is not supported."
            )

        operand_value = evaluate_math_node(node.operand)

        operation = ALLOWED_UNARY_OPERATORS[operator_type]

        return operation(operand_value)

    raise ValueError(
        "The mathematical expression is invalid."
    )


def extract_math_expression(question: str) -> str:
    """
    Extracts a mathematical expression from a user question.

    Examples:
    What is 25 * 18?        becomes 25 * 18
    Calculate 10 plus 5     becomes 10 + 5
    20 divided by 4         becomes 20 / 4
    """

    expression = question.lower().strip()

    replacements = {
        "multiplied by": "*",
        "multiply by": "*",
        "times": "*",
        "divided by": "/",
        "divide by": "/",
        "plus": "+",
        "minus": "-",
        "modulo": "%",
        "mod": "%",
    }

    for phrase, symbol in replacements.items():
        expression = expression.replace(
            phrase,
            symbol
        )

    expression = expression.replace("×", "*")
    expression = expression.replace("÷", "/")

    expression = re.sub(
        r"(?<=\d)\s*[xX]\s*(?=\d)",
        "*",
        expression
    )

    prefixes = [
        "please calculate",
        "calculate",
        "what is",
        "what's",
        "solve",
        "find",
    ]

    for prefix in prefixes:
        if expression.startswith(prefix):
            expression = expression[
                len(prefix):
            ].strip()

            break

    expression = expression.rstrip(" ?.")

    return expression


def calculate_math(question: str) -> str:
    """
    Safely calculates a mathematical expression.

    This function does not use Python eval().
    """

    try:
        expression = extract_math_expression(
            question
        )

        if not re.search(r"\d", expression):
            return (
                "No numbers were found in the "
                "math question."
            )

        if not re.search(
            r"[+\-*/%]",
            expression
        ):
            return (
                "No mathematical operator was found."
            )

        if not re.fullmatch(
            r"[\d\s+\-*/%.()]+",
            expression
        ):
            return (
                "I could not understand the "
                "mathematical expression."
            )

        parsed_expression = ast.parse(
            expression,
            mode="eval"
        )

        result = evaluate_math_node(
            parsed_expression
        )

        return str(result)

    except ZeroDivisionError:
        return (
            "A number cannot be divided by zero."
        )

    except Exception as error:
        return (
            "The mathematical expression could not "
            f"be calculated. Error: {error}"
        )


# ============================================================
# KNOWLEDGE BASE TOOL
# ============================================================

def search_knowledge_base(question: str) -> str:
    """
    Searches the local ChromaDB knowledge base using
    an embedding generated by nomic-embed-text.
    """

    if not CHROMA_PATH.exists():
        return (
            "The local knowledge database does not exist. "
            "Run python agentic_ai\\build_db.py first."
        )

    try:
        client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        try:
            collection = client.get_collection(
                name=COLLECTION_NAME
            )

        except Exception:
            return (
                "The ChromaDB collection agentic_kb "
                "does not exist. Run "
                "python agentic_ai\\build_db.py first."
            )

        embedding_response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=question
        )

        embeddings = embedding_response.get(
            "embeddings",
            []
        )

        if not embeddings:
            return (
                "The embedding model did not return "
                "an embedding for the question."
            )

        query_embedding = embeddings[0]

        collection_count = collection.count()

        if collection_count == 0:
            return (
                "The local knowledge base is empty. "
                "Add information to knowledge.txt and "
                "run build_db.py again."
            )

        number_of_results = min(
            3,
            collection_count
        )

        results = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=number_of_results,
            include=[
                "documents",
                "distances",
                "metadatas",
            ]
        )

        documents = results.get(
            "documents",
            []
        )

        distances = results.get(
            "distances",
            []
        )

        if (
            not documents
            or not documents[0]
            or not distances
            or not distances[0]
        ):
            return (
                "The knowledge base does not contain "
                "relevant information."
            )

        selected_documents = []

        for document, distance in zip(
            documents[0],
            distances[0]
        ):
            if (
                document
                and distance is not None
                and distance <= 1.0
            ):
                selected_documents.append(
                    document.strip()
                )

        if not selected_documents:
            return (
                "The knowledge base does not contain "
                "enough relevant information."
            )

        unique_documents = []

        for document in selected_documents:
            if document not in unique_documents:
                unique_documents.append(document)

        return "\n\n".join(
            unique_documents
        )

    except ConnectionError:
        return (
            "Could not connect to Ollama. Ensure that "
            "Ollama is running."
        )

    except Exception as error:
        return (
            "Knowledge-base search failed. "
            f"Error: {error}"
        )