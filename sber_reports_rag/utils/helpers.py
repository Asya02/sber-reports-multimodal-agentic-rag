from typing import Annotated, TypedDict

import tiktoken
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import add_messages

TOKENIZER = tiktoken.get_encoding("cl100k_base")

TAVILY_SEARCH_TOOL = TavilySearchResults(max_results=3)
LLM = ChatOpenAI(model="gpt-4o-mini")

MAX_RETRIES = 3
VERBOSE = True


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    documents: list[Document]
    candidate_answer: str
    retries: int
    web_fallback: bool


class GraphConfig(TypedDict):
    max_retries: int
