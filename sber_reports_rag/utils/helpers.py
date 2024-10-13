from typing import Annotated, NotRequired, TypedDict

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
    messages: NotRequired[Annotated[list[BaseMessage], add_messages]]
    question: NotRequired[str]
    documents: NotRequired[list[Document]]
    candidate_answer: NotRequired[str]
    retries: NotRequired[int]
    web_fallback: NotRequired[bool]


class GraphConfig(TypedDict):
    max_retries: NotRequired[int]
