import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import requests
from dotenv import load_dotenv,find_dotenv

def reload_dotenv():
    
    for key, value in os.environ.items():
        if key.startswith(""):  
            del os.environ[key]

   
    load_dotenv()

load_dotenv()
reload_dotenv()

GOOGLE_GEMINI_API_KEY=os.getenv("GEMINI_API")
os.environ['GOOGLE_GEMINI_API_KEY'] = GOOGLE_GEMINI_API_KEY # Replace with your actual API key

# Step 1: Load PDF and extract text
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Step 2: Split text into chunks
def split_text(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

# Load and process PDF
documents = load_pdf('BharathShanmugamResume_Cover_Letter.pdf')
texts = split_text(documents)

# Step 3: Use HuggingFaceBgeEmbeddings to generate embeddings
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Step 4: Create a Qdrant vector store
url = "http://localhost:6333"
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="vector_db"
)

print("Vector DB Successfully Created!")

# Step 5: Define a function to call the Google Gemini API
def query_gemini_api(prompt):
    api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    endpoint = "https://gemini.googleapis.com/v1/chat:completions"  # Replace with the actual Gemini API endpoint
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "gemini",  # Use the appropriate model name
        "temperature": 0.7
    }
    
    response = requests.post(endpoint, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

# Step 6: Define state and graph builder for LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    user_query = state["messages"][-1]['content']  # Extract user's last message (the query)

    # Generate the embedding for the user's query
    query_embedding = embeddings.embed_query(user_query)

    # Query Qdrant for similar documents
    search_results = qdrant.similarity_search(query_embedding, top_k=3)

    if search_results:
        # Retrieve the most relevant text from search results
        pdf_text_response = search_results[0].page_content

        # Use the Google Gemini API to generate a response based on the retrieved text
        response = query_gemini_api(f"Based on the following text: {pdf_text_response}, answer: {user_query}")
        return {"messages": [response]}
    else:
        return {"messages": ["No relevant information found."]}

# Add chatbot function to graph
graph_builder.add_node("chatbot", chatbot)

# Create the graph and compile it
graph = graph_builder.compile()

# Step 7: Start chat
def start_chat():
    print("You can start asking questions about the PDF. Type 'exit' or 'quit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        # Pass user input to the graph
        events = graph.stream(
            {"messages": [{"role": "user", "content": user_input}]}, stream_mode="values"
        )

        # Print responses
        for event in events:
            print("Bot:", event["messages"][-1])

start_chat()


# # pip install langchain langgraph

# from typing import Annotated, TypedDict
# from typing_extensions import Literal
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# from langchain.chat_models import C

# # Define the state of your chatbot
# class State(TypedDict):
#     messages: Annotated[list, add_messages]

# # Initialize the graph builder
# graph_builder = StateGraph(State)

# # Create a chatbot node
# def chatbot(state: State):
#     llm = ChatOpenAI(model_name="gpt-3.5-turbo")
#     response = llm.invoke(state["messages"])
#     return {"messages": [response]}

# # Add the chatbot node to the graph
# graph_builder.add_node("chatbot", chatbot)

# # Define the edges of the graph
# graph_builder.add_edge(START, "chatbot")
# graph_builder.add_edge("chatbot", END)

# # Compile the graph
# graph = graph_builder.compile()

# # Chat with the bot
# def chat_loop():
#     state = {"messages": []}
#     while True:
#         user_input = input("You: ")
#         state["messages"].append({"type": "human", "text": user_input})
#         for event in graph.stream_updates(state):
#             if event.type == "end":
#                 print("Bot:", event.outputs["messages"][0].text)
#                 state["messages"].append(event.outputs["messages"][0])

# if __name__ == "__main__":
#     chat_loop()