import pdfplumber
from qdrant_client import QdrantClient, models
from langchain.embeddings import HuggingFaceBgeEmbeddings

# Step 1: Load PDF and extract text
def extract_pdf_text(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''.join([page.extract_text() for page in pdf.pages])
    return text

pdf_text = extract_pdf_text('BharathShanmugamResume_Cover_Letter.pdf')

# Step 2: Split text into chunks
def split_text(text, chunk_size=500, chunk_overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

texts = split_text(pdf_text)

# Step 3: Use HuggingFaceBgeEmbeddings to generate embeddings
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Generate embeddings for each text chunk
embeddings_list = [embeddings.embed_query(chunk) for chunk in texts]

# Step 4: Initialize Qdrant client and create collection
client = QdrantClient(host="localhost", port=6333)

# Create collection with vector size of 1024 (adjust this if your model output size is different)
client.recreate_collection(
    collection_name="pdf_collection",
    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
)

# Step 5: Insert points into Qdrant
points = [
    {
        "id": idx,
        "vector": embedding,
        "payload": {"text": text}
    }
    for idx, (embedding, text) in enumerate(zip(embeddings_list, texts))
]

client.upsert(collection_name="pdf_collection", points=points)

# Step 6: Define query function
def query_pdf(query):
    # Generate embedding for the query
    query_embedding = embeddings.embed_query(query)

    # Query Qdrant
    search_results = client.search(
        collection_name="pdf_collection",
        query_vector=query_embedding,
        limit=3
    )

    # Return the most relevant text
    if search_results:
        return search_results[0].payload["text"]
    else:
        return "No relevant information found."

# Step 7: Start Chat
def start_chat():
    print("You can start asking questions about the PDF. Type 'exit' or 'quit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        response = query_pdf(user_input)
        print("Bot:", response)

start_chat()







# from langchain.vectorstores import Qdrant
# from langchain.embeddings import HuggingFaceBgeEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader

# loader = PyPDFLoader("BharathShanmugamResume_Cover_Letter.pdf")
# documents = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
#                                                    chunk_overlap=50)
# texts = text_splitter.split_documents(documents)

# # Load the embedding model 
# model_name = "BAAI/bge-large-en"
# model_kwargs = {'device': 'cpu'}
# encode_kwargs = {'normalize_embeddings': False}
# embeddings = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )

# url = "http://localhost:6333"
# qdrant = Qdrant.from_documents(
#     texts,
#     embeddings,
#     url=url,
#     prefer_grpc=False,
#     collection_name="vector_db"
# )

# print("Vector DB Successfully Created!")






