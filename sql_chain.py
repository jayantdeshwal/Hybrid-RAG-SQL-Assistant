"""
LangChain SQL workflow using:
- Groq chat model
- SQLite database
- LangChain SQL query generation

This file owns the complete database-question pipeline.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_classic.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


load_dotenv()

DATABASE_PATH = Path("data/database/app.db")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def get_llm():
    """Create the Groq model used by the SQL chain."""

    return ChatGroq(model=DEFAULT_GROQ_MODEL, temperature=0)


def get_database(database_path: Optional[Path] = None):
    """Connect to the local SQLite database."""

    database_path = database_path or DATABASE_PATH

    if not database_path.exists():
        raise FileNotFoundError(f"Database not found at {database_path}.")

    return SQLDatabase.from_uri(f"sqlite:///{database_path}")


def answer_data_question(
    question: str,
    database_path: Optional[Path] = None,
) -> dict:
    """
    Answer a structured data question in three steps:
    1. generate SQL
    2. execute SQL
    3. convert the raw result into a readable answer
    """

    db = get_database(database_path=database_path)
    llm = get_llm()

    sql_chain = create_sql_query_chain(llm, db)
    execute_query = QuerySQLDatabaseTool(db=db)

    sql_query = sql_chain.invoke({"question": question})
    sql_result = execute_query.invoke(sql_query)

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a data analyst. Explain the SQL result in clear, "
                    "simple business language."
                ),
            ),
            (
                "human",
                (
                    "Question: {question}\n\n"
                    "SQL Query: {sql_query}\n\n"
                    "SQL Result: {sql_result}\n\n"
                    "Write the final answer for the user."
                ),
            ),
        ]
    )

    answer_chain = answer_prompt | llm
    final_answer = answer_chain.invoke(
        {
            "question": question,
            "sql_query": sql_query,
            "sql_result": sql_result,
        }
    )

    return {
        "answer": final_answer.content,
        "sql_query": sql_query,
    }
