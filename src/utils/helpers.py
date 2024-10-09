import tiktoken
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

from backend.rag import get_retriever

TOKENAZER = tiktoken.get_encoding("cl100k_base")

tavily_search_tool = TavilySearchResults(max_results=3)
retriever = get_retriever()
llm = ChatOpenAI(model="gpt-4o-mini")
