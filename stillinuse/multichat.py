# import os 
# from langchain.document_loaders import PyPDFLoader 
# from langchain.schema import Document 
# from langchain.vectorstores import FAISS 
# from langchain_google_genai import GoogleGenerativeAIEmbeddings 
# from typing import TypedDict, List 
# from Langgraph.graph import StateGraph 
# from langgraph.prebuilt import ToolNode 
# from langchain.document_loaders import PyPDFLoader 
# from langchain.schema import Document 
# from langchain.vectorstores import FAISS 
# from Langchain.prompts import PromptTemplate 
# from langchain.tools import Tool 
# from langchain_google_genai import ChatGoogleGenerativeAl 
# from langchain.text_splitter import RecursiveCharacterTextSplitter 
# from langchain.llms import OpenAI @title 



from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)