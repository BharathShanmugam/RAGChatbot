from typing import Annotated, List
from typing_extensions import TypedDict
import pdfplumber
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from langgraph.graph import StateGraph, START, END, add_messages
import pdfplumber
from langchain_core.messages import BaseMessage



# Define the state schema
class State(TypedDict):
    messages: List[BaseMessage]

# Function to extract text from PDF
def extract_pdf_text(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''.join([page.extract_text() for page in pdf.pages])
    return text

# Function to split text into chunks
def split_text(text, chunk_size=500, chunk_overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Load and process the PDF
pdf_text = extract_pdf_text('Document 1.pdf')
texts = split_text(pdf_text)

# Load the embedding model
model = SentenceTransformer('BAAI/bge-large-en')
embeddings = [model.encode(chunk) for chunk in texts]

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Check if collection exists and create if it doesn't
if not client.collection_exists("new_collection"):
    client.create_collection(
        collection_name="new_collection",
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
    )

# Insert points into Qdrant
points = [
    models.PointStruct(
        id=idx,
        vector=embedding.tolist(),
        payload={"text": text}
    )
    for idx, (embedding, text) in enumerate(zip(embeddings, texts))
]

client.upsert(collection_name="new_collection", points=points)

# Function to query the PDF using Qdrant
def query_pdf(query):
    query_embedding = model.encode(query)
    search_results = client.search(
        collection_name="new_collection",
        query_vector=query_embedding.tolist(),
        limit=3
    )

    if search_results:
        return search_results[0].payload["text"]
    else:
        return "No relevant information found."

# Initialize the StateGraph
graph = StateGraph(State)

# Function to handle chatbot logic
def chatbot(state: State):
    user_query = state["messages"][-1]["content"]
    response_text = query_pdf(user_query)
    
    # Ensure to format the assistant's response correctly
    print("before state:", state)

    state["messages"].append({"role": "assistant", "content": response_text})
    print("Current state:", state)

    return state



# Start the interactive chat loop
def start_chat():
    print("You can start asking questions about the PDF. Type 'exit' or 'quit' to end the chat.\n")
    
    # Initialize the state for the conversation
    state: State = {"messages": [{"role": "system", "content": "Chat started."}]}

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            state["messages"].append({"role": "system", "content": "Chat ended."})
            print("Exiting chat...")
            break

        # Ensure to add user input correctly
        state["messages"].append({"role": "user", "content": user_input})
        
        # Generate response
        updated_state = chatbot(state)

        # Print the assistant's response
        print("Assistant:", updated_state["messages"][-1]["content"])

# Run the chat
start_chat()








































# import pdfplumber
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient, models
# from langgraph.graph import Graph

# # Function to extract text from PDF
# def extract_pdf_text(file_path):
#     with pdfplumber.open(file_path) as pdf:
#         text = ''.join([page.extract_text() for page in pdf.pages])
#     return text

# # Function to split text into chunks
# def split_text(text, chunk_size=500, chunk_overlap=50):
#     chunks = []
#     for i in range(0, len(text), chunk_size - chunk_overlap):
#         chunks.append(text[i:i + chunk_size])
#     return chunks

# # Load and process the PDF
# pdf_text = extract_pdf_text('BharathShanmugamResume_Cover_Letter.pdf')
# texts = split_text(pdf_text)

# # Load the embedding model
# model = SentenceTransformer('BAAI/bge-large-en')
# embeddings = [model.encode(chunk) for chunk in texts]

# # Initialize Qdrant client
# client = QdrantClient(host="localhost", port=6333)

# # Check if collection exists and create if it doesn't
# if not client.collection_exists("new_collection"):
#     client.create_collection(
#         collection_name="pdf_collection",
#         vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
#     )

# # Insert points into Qdrant
# points = [
#     models.PointStruct(
#         id=idx,
#         vector=embedding.tolist(),
#         payload={"text": text}
#     )
#     for idx, (embedding, text) in enumerate(zip(embeddings, texts))
# ]

# client.upsert(collection_name="pdf_collection", points=points)

# # Initialize the Graph for manual querying
# graph = Graph()

# # Function to query the PDF using Qdrant
# def query_pdf(query):
#     # Generate the embedding for the query using the same embedding model
#     query_embedding = model.encode([query])[0]

#     # Query Qdrant directly using the Qdrant client
#     search_results = client.search(
#         collection_name="pdf_collection",
#         query_vector=query_embedding,
#         limit=3
#     )

#     # Process the results and return the most relevant text
#     if search_results:
#         return search_results[0].payload["text"]
#     else:
#         return "No relevant information found."

# # Function to handle chatbot logic
# def chatbot(state: dict):
#     user_query = state["messages"][-1]["content"]  # Extract user's last message (the query)
#     response = query_pdf(user_query)  # Query the PDF
#     return {"messages": state["messages"] + [{"role": "assistant", "content": response}]}

# # Start the interactive chat loop
# def start_chat():
#     print("You can start asking questions about the PDF. Type 'exit' or 'quit' to end the chat.\n")
    
#     while True:
#         user_input = input("You: ")

#         if user_input.lower() in ["exit", "quit"]:
#             print("Exiting chat...")
#             break

#         # Simulate conversation state
#         state = {"messages": [{"role": "user", "content": user_input}]}
        
#         # Generate response
#         response = chatbot(state)

#         # Print the assistant's response
#         print("Assistant:", response["messages"][-1]["content"])

# # Run the chat
# start_chat()


































# import pdfplumber
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient, models

# # Step 1: Extract text from the PDF
# def extract_pdf_text(file_path):
#     with pdfplumber.open(file_path) as pdf:
#         text = ''.join([page.extract_text() for page in pdf.pages])
#     return text

# pdf_text = extract_pdf_text('BharathShanmugamResume_Cover_Letter.pdf')

# # Step 2: Split the text into chunks
# def split_text(text, chunk_size=500, chunk_overlap=50):
#     chunks = []
#     for i in range(0, len(text), chunk_size - chunk_overlap):
#         chunks.append(text[i:i + chunk_size])
#     return chunks

# texts = split_text(pdf_text)

# # Step 3: Generate embeddings using Sentence-Transformers
# model = SentenceTransformer('BAAI/bge-large-en')
# embeddings = [model.encode(chunk) for chunk in texts]

# # Step 4: Initialize Qdrant client
# client = QdrantClient(host="localhost", port=6333)

# # Step 5: Check if the collection already exists
# collection_name = "pdf_collection"
# if not client.collection_exists(collection_name=collection_name):
#     # If the collection does not exist, create it
#     client.create_collection(
#         collection_name=collection_name,
#         vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)  # Adjust size according to the model
#     )

# # Step 6: Insert embeddings and texts as points into Qdrant
# points = [
#     models.PointStruct(
#         id=idx,
#         vector=embedding,
#         payload={"text": text}
#     )
#     for idx, (embedding, text) in enumerate(zip(embeddings, texts))
# ]

# client.upsert(collection_name=collection_name, points=points)

# # Step 7: Define the query function to search the vector database
# def query_pdf(query):
#     # Generate the embedding for the query using the same model
#     query_embedding = model.encode([query])[0]

#     # Query Qdrant for the most similar vectors
#     search_results = client.search(
#         collection_name=collection_name,
#         query_vector=query_embedding,
#         limit=3
#     )

#     # Return the most relevant result
#     if search_results:
#         return search_results[0].payload["text"]  # Access the payload using dot notation
#     else:
#         return "No relevant information found."

# # Example usage
# response = query_pdf("What is the main idea of the document?")
# print(response)










# import pdfplumber
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient, models

# # Step 1: Extract text from the PDF
# def extract_pdf_text(file_path):
#     with pdfplumber.open(file_path) as pdf:
#         text = ''.join([page.extract_text() for page in pdf.pages])
#     return text

# pdf_text = extract_pdf_text('BharathShanmugamResume_Cover_Letter.pdf')

# # Step 2: Split the text into chunks
# def split_text(text, chunk_size=500, chunk_overlap=50):
#     chunks = []
#     for i in range(0, len(text), chunk_size - chunk_overlap):
#         chunks.append(text[i:i + chunk_size])
#     return chunks

# texts = split_text(pdf_text)

# # Step 3: Generate embeddings using Sentence-Transformers
# model = SentenceTransformer('BAAI/bge-large-en')
# embeddings = [model.encode(chunk) for chunk in texts]

# # Step 4: Initialize Qdrant client and create a collection
# client = QdrantClient(host="localhost", port=6333)

# # Correct method to create a collection with vector configuration
# client.recreate_collection(
#     collection_name="pdf_collection",
#     vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)  # Adjust size according to model
# )

# # Step 5: Insert embeddings and texts as points into Qdrant
# points = [
#     models.PointStruct(
#         id=idx,
#         vector=embedding,
#         payload={"text": text}
#     )
#     for idx, (embedding, text) in enumerate(zip(embeddings, texts))
# ]

# client.upsert(collection_name="pdf_collection", points=points)

# # Step 6: Define the query function to search the vector database
# def query_pdf(query):
#     # Generate the embedding for the query using the same model
#     query_embedding = model.encode([query])[0]

#     # Query Qdrant for the most similar vectors
#     search_results = client.search(
#         collection_name="pdf_collection",
#         query_vector=query_embedding,
#         limit=3
#     )

#     # Return the most relevant result
#     if search_results:
#         return search_results[0].payload["text"]  # Access the payload using dot notation
#     else:
#         return "No relevant information found."

# # Example usage
# response = query_pdf("who cover letter is this ")
# print(response)






















