import os
import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_community.llms.openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient,models
from sentence_transformers import SentenceTransformer
from typing import List
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Set your API keys here
folder_path = "pdf"
openai_api_key = os.getenv()
qdrant_host = "localhost"  
qdrant_port = 6333
qdrant_collection = "pdf_chunks_groq"


def reload_dotenv():
    
    for key, value in os.environ.items():
        if key.startswith(""):  
            del os.environ[key]

   
    load_dotenv()

load_dotenv()
reload_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

# Initialize Qdrant client
client = QdrantClient(host=qdrant_host, port=qdrant_port)

older_path = "pdf"
pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]

# Function to extract text from PDF files
def extract_pdf_text(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''.join([page.extract_text() for page in pdf.pages])
    return text

# Load and extract text from each PDF file
textss = []
for pdf_file in pdf_files:
    text = extract_pdf_text(pdf_file)
    textss.append(text)

# Create a list of documents from the extracted text
documents = [Document(page_content=text) for text in textss]


# Split documents into chunks for processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Load the embedding model (ensure it's the correct model for 1024 dimensions)
sentence_model = SentenceTransformer('BAAI/bge-large-en')  # Ensure this model returns 1024-dim vectors

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Check if collection exists and create if it doesn't
qdrant_collection = "new_collection"
if not client.collection_exists(qdrant_collection):
    client.create_collection(
        collection_name=qdrant_collection,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE) 
    )

# Insert points into Qdrant
points = [
    models.PointStruct(
        id=idx,
        vector=sentence_model.encode(doc.page_content).tolist(),
        payload={"text": doc.page_content}
    )
    for idx, doc in enumerate(docs)
]

client.upsert(collection_name=qdrant_collection, points=points)


# Initialize OpenAI LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY,model_name="gemma-7b-it")

def search_vector_db(query: str, k: int = 3) -> List[str]:
    """Search the vector database for relevant documents."""
    query_vector = sentence_model.encode(query).tolist()
    search_results = client.search(
        collection_name=qdrant_collection,
        query_vector=query_vector,
        limit=k
    )
    return [result.payload["text"] for result in search_results]

def summarize_context(context: str) -> str:
    """Provide a summary of the retrieved context."""
    prompt = PromptTemplate(template="Summarize the following context to provide a concise overview: {context}")
    summary = llm.invoke(prompt.format(context=context))  # Use invoke() instead of __call__()
    
    # Access the content of the returned AIMessage object
    return summary.content.strip()  # Use summary.content instead of just summary




def generate_response(context: str, question: str) -> str:
    """Generate a response based on the context and question."""
    prompt = PromptTemplate(template="You are a helpful assistant for a user guide. Question: {question} Context: {context} Answer: <>")
    formatted_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(formatted_prompt)  # Use invoke() instead of __call__()
    
    # Access the content of the returned AIMessage object
    return response.content.strip()  # Use response.content instead of just response



# Define the state expansion, search, summarization, and response generation functions
def expand(state: dict) -> dict:
    """Expand the query and store it in the state."""
    state["expanded_query"] = state["question"]
    return state

def search(state: dict) -> dict:
    """Search for relevant documents based on the expanded query."""
    results = search_vector_db(state["expanded_query"])
    state["context"] = results
    return state

def summarize(state: dict) -> dict:
    """Summarize the retrieved context."""
    context = "".join(state["context"]) if state["context"] else ""
    state["summarized_context"] = summarize_context(context)
    return state

def generate(state: dict) -> dict:
    """Generate a response based on the summarized context and the question."""
    response = generate_response(state["summarized_context"], state["question"])
    state["response"] = response
    return state

# Create the graph workflow
workflow = StateGraph(dict)
workflow.add_node("expand", expand)
workflow.add_node("search", search)
workflow.add_node("summarize", summarize)
workflow.add_node("generate", generate)

# Define the edges of the graph
workflow.set_entry_point("expand")
workflow.add_edge("expand", "search")
workflow.add_edge("search", "summarize")
workflow.add_edge("summarize", "generate")
workflow.set_finish_point("generate")

# Compile the graph
graph = workflow.compile()

def run_graph(question: str):   
    """Run the graph with the given question."""
    input_state = {"question": question}  
    result = graph.invoke(input=input_state)  # Pass the dictionary to the graph
    return result["response"]

if __name__ == "__main__":
    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() in ["exit", "quit"]:
            break
        response = run_graph(question)
        print("Response:", response)




    # # Get the image data
    # from PIL import Image
    # import io

    # try:
    #     # Get the image data from the graph (as a PNG)
    #     image_data = graph.get_graph().draw_mermaid_png()

    #     # Save the image to a file
    #     image_path = "graph_image.png"  # Define where you want to save the image
    #     with open(image_path, 'wb') as f:
    #         f.write(image_data)

    #     print(f"Image saved successfully to {image_path}")

    # except Exception as e:
    #     print(f"An error occurred: {e}")

















# Example usage
# if __name__ == "__main__":
#     while True:
#         question = "Compare the features of smart tv and Laptop"
#         if question.lower() in ["exit", "quit"]:
            
#             response = run_graph(question)
#             print("Response:", response)
#             break

