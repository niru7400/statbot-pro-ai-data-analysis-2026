import os
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama import ChatOllama


BLOCKED_TERMS = [
    "os.system",
    "subprocess",
    "rm -rf",
    "eval(",
    "exec(",
    "__import__",
    "open(",
    "write(",
    "delete",
    "remove",
    "format",
]


def is_safe_question(question: str) -> bool:
    q = question.lower()
    return not any(term in q for term in BLOCKED_TERMS)


def create_agent(csv_file: str):
    df = pd.read_csv(csv_file)

    os.makedirs("charts", exist_ok=True)

    llm = ChatOllama(
        model="llama3",
        temperature=0
    )

    prefix = """
You are StatBot Pro, an autonomous CSV data analyst working with a pandas dataframe called df.

Rules:
- Use python_repl_ast tool to run python code.
- Use pandas for calculations and dataframe analysis.
- Answer only the user's current question.
- Do not ask extra questions.
- Do not continue to another question by yourself.
- Return only the final answer.
- Never use OS/system commands or unsafe code.
"""

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        allow_dangerous_code=True,
        max_iterations=10,
        prefix=prefix
    )

    return agent