import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.llms import openai

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

# Step 5: Set up OpenAI API key and initialize the LLM
os.environ['OPENAI_API_KEY'] = 'your-openai-api-key'  # Replace with your actual API key
llm = OpenAI(model="text-davinci-003", temperature=0.7)

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

        # Use OpenAI LLM to generate a response based on the retrieved text
        response = llm.run(f"Based on the following text, {pdf_text_response}, answer: {user_query}")
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
