"""
LangChain RAG workflow using:
- Hugging Face embeddings
- FAISS vector store
- Groq chat model

This file owns the complete document-question pipeline.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

DOCS_DIR = Path("data/docs")
VECTOR_STORE_DIR = Path("data/vectorstore")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5",
)


def get_llm():
    """Create the Groq chat model used by the RAG chain."""

    return ChatGroq(model=DEFAULT_GROQ_MODEL, temperature=0)


def get_embeddings():
    """
    Create Hugging Face embeddings.

    `normalize_embeddings=True` is a good default for similarity search.
    """

    return HuggingFaceEmbeddings(
        model_name=DEFAULT_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents(docs_dir: Optional[Path] = None):
    """Load supported files from the documents folder."""

    docs_dir = docs_dir or DOCS_DIR

    text_loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    pdf_loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )

    documents = []
    if docs_dir.exists():
        documents.extend(text_loader.load())
        documents.extend(pdf_loader.load())

    return documents


def build_vector_store(
    docs_dir: Optional[Path] = None,
    vector_store_dir: Optional[Path] = None,
):
    """
    Build a FAISS index from the documents folder and save it locally.

    This is the preprocessing step of the RAG pipeline:
    raw documents -> chunks -> embeddings -> FAISS index
    """

    docs_dir = docs_dir or DOCS_DIR
    vector_store_dir = vector_store_dir or VECTOR_STORE_DIR

    documents = load_documents(docs_dir=docs_dir)
    if not documents:
        raise ValueError(f"No documents found in {docs_dir}.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    vector_store = FAISS.from_documents(chunks, get_embeddings())
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(vector_store_dir))


def load_vector_store(vector_store_dir: Optional[Path] = None):
    """Load the saved FAISS index from disk."""

    vector_store_dir = vector_store_dir or VECTOR_STORE_DIR

    if not vector_store_dir.exists():
        raise FileNotFoundError(
            "Vector store not found. Use the sidebar button to build it first."
        )

    return FAISS.load_local(
        str(vector_store_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def answer_document_question(
    question: str,
    vector_store_dir: Optional[Path] = None,
) -> dict:
    """
    Run the full RAG workflow and return answer plus sources.

    Retrieval chain flow:
    question -> retriever -> relevant chunks -> prompt -> LLM -> final answer
    """

    vector_store = load_vector_store(vector_store_dir=vector_store_dir)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a document question-answering assistant. "
                    "Answer only from the retrieved context. "
                    "If the context is not enough, say so clearly."
                ),
            ),
            (
                "human",
                (
                    "Question: {input}\n\n"
                    "Retrieved Context:\n{context}\n\n"
                    "Return a grounded answer."
                ),
            ),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    response = rag_chain.invoke({"input": question})

    sources = sorted(
        {
            Path(document.metadata.get("source", "unknown")).name
            for document in response.get("context", [])
        }
    )

    return {
        "answer": response["answer"],
        "sources": sources,
    }
