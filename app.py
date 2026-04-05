"""
Main Streamlit app for the hybrid RAG + SQL assistant.

This file keeps the top-level flow in one place so it is easy to follow:
1. Accept a user question
2. Route the question
3. Call either the RAG workflow or the SQL workflow
4. Display the answer
"""

import os
from pathlib import Path
from typing import Literal

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from rag_chain import answer_document_question, build_vector_store
from sql_chain import answer_data_question


load_dotenv()

DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_DOCS_DIR = Path("data/docs")
DEFAULT_VECTOR_STORE_DIR = Path("data/vectorstore")
DEFAULT_DATABASE_PATH = Path("data/database/app.db")


class RouteDecision(BaseModel):
    """Structured router output."""

    route: Literal["rag", "sql"] = Field(
        ...,
        description="The workflow that should answer the question.",
    )
    reason: str = Field(
        ...,
        description="Why this route was selected.",
    )


def inject_custom_styles():
    """Add lightweight custom styling for a cleaner sidebar and page."""

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top left, #fff1dc 0%, transparent 38%),
                linear-gradient(180deg, #fffaf0 0%, #f5efe3 100%);
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
        }
        .sidebar-hero {
            background: linear-gradient(135deg, #173b2f 0%, #2d6a4f 100%);
            color: #ffffff;
            padding: 1rem 1rem 0.9rem 1rem;
            border-radius: 18px;
            margin-bottom: 1rem;
            box-shadow: 0 12px 24px rgba(23, 59, 47, 0.16);
        }
        .sidebar-hero h3 {
            margin: 0;
            font-size: 1.05rem;
        }
        .sidebar-hero p {
            margin: 0.35rem 0 0 0;
            font-size: 0.84rem;
            line-height: 1.4;
            color: rgba(255, 255, 255, 0.86);
        }
        .status-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(23, 59, 47, 0.08);
            border-radius: 16px;
            padding: 0.85rem 0.9rem;
            margin-bottom: 0.65rem;
            box-shadow: 0 6px 18px rgba(23, 59, 47, 0.06);
        }
        .status-label {
            font-size: 0.76rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #5f6f68;
            margin-bottom: 0.2rem;
        }
        .status-value {
            font-size: 0.98rem;
            font-weight: 700;
            color: #173b2f;
        }
        .sidebar-section {
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid rgba(23, 59, 47, 0.08);
            border-radius: 18px;
            padding: 0.9rem 0.9rem 0.5rem 0.9rem;
            margin: 0.8rem 0 1rem 0;
        }
        .sidebar-section h4 {
            margin: 0 0 0.4rem 0;
            color: #173b2f;
            font-size: 0.98rem;
        }
        .sidebar-section p {
            margin: 0.2rem 0 0.5rem 0;
            color: #50605a;
            font-size: 0.84rem;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_status_card(label: str, value: str):
    """Render a small styled status card in the sidebar."""

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">{label}</div>
            <div class="status-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_router_chain():
    """
    Build a small routing chain.

    We use the model in structured-output mode so the router returns a
    predictable object instead of free-form text.
    """

    llm = ChatGroq(model=DEFAULT_GROQ_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(RouteDecision)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a router for a hybrid AI app. "
                    "Choose 'sql' when the question is about structured data, "
                    "metrics, aggregations, trends, filters, sales, revenue, "
                    "counts, tables, or database records. "
                    "Choose 'rag' when the question is about policies, "
                    "documents, manuals, contracts, FAQs, or file-based knowledge."
                ),
            ),
            ("human", "Question: {question}"),
        ]
    )

    return prompt | structured_llm


def decide_route(question: str) -> RouteDecision:
    """Route the question to either the RAG flow or the SQL flow."""

    cleaned_question = question.strip()
    if not cleaned_question:
        return RouteDecision(
            route="rag",
            reason="The question was empty, so the app defaulted to document search.",
        )

    router_chain = get_router_chain()
    return router_chain.invoke({"question": cleaned_question})


def initialize_session_state():
    """Create chat history storage when the app runs for the first time."""

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "docs_dir" not in st.session_state:
        st.session_state.docs_dir = str(DEFAULT_DOCS_DIR)
    if "vector_store_dir" not in st.session_state:
        st.session_state.vector_store_dir = str(DEFAULT_VECTOR_STORE_DIR)
    if "database_path" not in st.session_state:
        st.session_state.database_path = str(DEFAULT_DATABASE_PATH)


def save_uploaded_documents(uploaded_files) -> list[str]:
    """Save uploaded documents into the app's docs folder."""

    DEFAULT_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for uploaded_file in uploaded_files:
        file_path = DEFAULT_DOCS_DIR / uploaded_file.name
        file_path.write_bytes(uploaded_file.getbuffer())
        saved_files.append(uploaded_file.name)

    return saved_files


def save_uploaded_database(uploaded_file) -> str:
    """Save an uploaded SQLite database and make it the active database."""

    DEFAULT_DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_DATABASE_PATH.write_bytes(uploaded_file.getbuffer())
    return str(DEFAULT_DATABASE_PATH)


def get_status_summary():
    """Return small status flags for the current app resources."""

    docs_dir = Path(st.session_state.docs_dir)
    vector_store_dir = Path(st.session_state.vector_store_dir)
    database_path = Path(st.session_state.database_path)

    document_count = 0
    if docs_dir.exists():
        document_count = len(list(docs_dir.glob("*.txt"))) + len(list(docs_dir.glob("*.pdf")))

    vector_store_ready = (
        vector_store_dir.exists()
        and any(vector_store_dir.iterdir())
    )
    database_ready = database_path.exists()

    return {
        "document_count": document_count,
        "vector_store_ready": vector_store_ready,
        "database_ready": database_ready,
    }


def display_chat_history():
    """Render all previous user and assistant messages."""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                if message.get("route"):
                    st.caption(f"Route used: {message['route']}")
                if message.get("reason"):
                    st.caption(f"Why this route: {message['reason']}")
                if message.get("sources"):
                    st.markdown("**Sources**")
                    for source in message["sources"]:
                        st.write(f"- {source}")
                if message.get("sql_query"):
                    st.markdown("**Generated SQL**")
                    st.code(message["sql_query"], language="sql")


def main():
    """Run the Streamlit application."""

    st.set_page_config(page_title="Hybrid RAG + SQL Assistant", layout="wide")
    inject_custom_styles()
    st.title("Hybrid RAG + SQL Assistant")
    st.write(
        "Ask a document question or a database question. "
        "The app will route it to the correct workflow."
    )

    initialize_session_state()

    with st.sidebar:
        status = get_status_summary()

        st.markdown(
            """
            <div class="sidebar-hero">
                <h3>Workspace Control Panel</h3>
                <p>Upload your own files, switch to default data, and verify that
                the app is ready before asking questions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Status")
        render_status_card(
            "Documents",
            f"{'Ready' if status['document_count'] > 0 else 'Missing'} · {status['document_count']} file(s)",
        )
        render_status_card(
            "Vector Store",
            "Built" if status["vector_store_ready"] else "Not built",
        )
        render_status_card(
            "Database",
            "Active" if status["database_ready"] else "Missing",
        )

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("<h4>Configuration</h4>", unsafe_allow_html=True)
        st.write(f"Groq model: `{DEFAULT_GROQ_MODEL}`")
        st.write("Embedding model: `BAAI/bge-small-en-v1.5`")
        st.write(f"Active docs folder: `{st.session_state.docs_dir}`")
        st.write(f"Active database: `{st.session_state.database_path}`")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("<h4>Documents</h4>", unsafe_allow_html=True)
        st.markdown(
            (
                "<p>Upload your own text or PDF files, or keep using the default "
                "hospital documents already included in the app.</p>"
            ),
            unsafe_allow_html=True,
        )
        if st.button("Use Default Hospital Documents"):
            st.session_state.docs_dir = str(DEFAULT_DOCS_DIR)
            st.success("Default hospital documents are now active.")

        uploaded_docs = st.file_uploader(
            "Upload text or PDF files",
            type=["txt", "pdf"],
            accept_multiple_files=True,
        )
        if st.button("Save Uploaded Documents"):
            if not uploaded_docs:
                st.warning("Please upload at least one text or PDF file.")
            else:
                saved_files = save_uploaded_documents(uploaded_docs)
                st.session_state.docs_dir = str(DEFAULT_DOCS_DIR)
                st.success(f"Saved {len(saved_files)} document(s).")
        st.caption("Try a sample document question:")
        st.caption("- What is the visitor policy for ICU patients?")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("<h4>Database</h4>", unsafe_allow_html=True)
        st.markdown(
            (
                "<p>Upload your own SQLite database, or switch back to the default "
                "hospital database for a guided demo experience.</p>"
            ),
            unsafe_allow_html=True,
        )
        if st.button("Use Default Hospital Database"):
            st.session_state.database_path = str(DEFAULT_DATABASE_PATH)
            st.success("Default hospital database is now active.")

        uploaded_db = st.file_uploader(
            "Upload a SQLite database file",
            type=["db", "sqlite", "sqlite3"],
        )
        if st.button("Use Uploaded Database"):
            if uploaded_db is None:
                st.warning("Please upload a SQLite database file first.")
            else:
                st.session_state.database_path = save_uploaded_database(uploaded_db)
                st.success("Uploaded database is now active.")

        st.caption("Try a sample database question:")
        st.caption("- How many patients are registered from Delhi?")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Build / Refresh Vector Store"):
            with st.spinner("Building vector store from documents..."):
                build_vector_store(
                    Path(st.session_state.docs_dir),
                    Path(st.session_state.vector_store_dir),
                )
            st.success("Vector store is ready.")

    display_chat_history()

    question = st.chat_input("Ask a question")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Thinking..."):
        route_decision = decide_route(question)

        if route_decision.route == "rag":
            result = answer_document_question(
                question,
                Path(st.session_state.vector_store_dir),
            )
        else:
            result = answer_data_question(
                question,
                Path(st.session_state.database_path),
            )

    assistant_message = {
        "role": "assistant",
        "content": result["answer"],
        "route": route_decision.route,
        "reason": route_decision.reason,
        "sources": result.get("sources", []),
        "sql_query": result.get("sql_query"),
    }
    st.session_state.messages.append(assistant_message)

    with st.chat_message("assistant"):
        st.markdown(result["answer"])
        st.caption(f"Route used: {route_decision.route}")
        st.caption(f"Why this route: {route_decision.reason}")

        if result.get("sources"):
            st.markdown("**Sources**")
            for source in result["sources"]:
                st.write(f"- {source}")

        if result.get("sql_query"):
            st.markdown("**Generated SQL**")
            st.code(result["sql_query"], language="sql")


if __name__ == "__main__":
    main()
