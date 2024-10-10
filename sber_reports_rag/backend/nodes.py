from langchain_core.documents import Document
from langchain_core.messages import AIMessage, convert_to_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from sber_reports_rag.backend.rag import RETRIVER
from sber_reports_rag.utils.helpers import LLM, TAVILY_SEARCH_TOOL, GraphState
from sber_reports_rag.utils.templates import QUERY_REWRITING_TEMPLATE, RAG_TEMPLATE


def document_search(state: GraphState):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")

    question = convert_to_messages(state["messages"])[-1].content

    # Retrieval
    documents = RETRIVER.invoke(question)
    return {"documents": documents, "question": question, "web_fallback": True}


def generate(state: GraphState):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    retries = state["retries"] if state.get("retries") is not None else -1

    rag_prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    rag_chain = rag_prompt | LLM | StrOutputParser()
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"retries": retries + 1, "candidate_answer": generation}


def transform_query(state: GraphState):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print("---TRANSFORM QUERY---")

    question = state["question"]

    query_rewriting_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QUERY_REWRITING_TEMPLATE),
            (
                "human",
                "Вот исходный вопрос: \n\n {question} \n Сформулируйте улучшенный вопрос.",
            ),
        ]
    )
    # Re-write question
    query_rewriter = query_rewriting_prompt | LLM | StrOutputParser()
    better_question = query_rewriter.invoke({"question": question})
    return {"question": better_question}


def web_search(state: GraphState):
    print("---RUNNING WEB SEARCH---")

    question = state["question"]
    documents = state["documents"]
    search_results = TAVILY_SEARCH_TOOL.invoke(question)
    search_content = "\n".join([d["content"] for d in search_results])
    documents.append(
        Document(page_content=search_content, metadata={"source": "websearch"})
    )
    return {"documents": documents, "web_fallback": False}


def finalize_response(state: GraphState):
    print("---FINALIZING THE RESPONSE---")

    return {"messages": [AIMessage(content=state["candidate_answer"])]}
