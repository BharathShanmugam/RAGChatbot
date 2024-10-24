import pdfplumber

def extract_pdf_text(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''.join([page.extract_text() for page in pdf.pages])
    return text

pdf_text = extract_pdf_text('data.pdf')
def split_text(text, chunk_size=500, chunk_overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

texts = split_text(pdf_text)




from sentence_transformers import SentenceTransformer

# Load a pre-trained transformer model
model = SentenceTransformer('BAAI/bge-large-en')

# Generate embeddings for each chunk of text
embeddings = [model.encode(chunk) for chunk in texts]

from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# Create a collection in Qdrant with vector size based on your model (1024 dimensions for BAAI/bge-large-en)
client.recreate_collection(
    collection_name="pdf_collection",
    vector_size=1024,  # Change this according to your embedding model size
    distance="Cosine"
)

# Insert the embeddings along with their corresponding text into Qdrant
points = [
    {
        "id": idx,
        "vector": embedding,
        "payload": {"text": text}
    }
    for idx, (embedding, text) in enumerate(zip(embeddings, texts))
]

client.upsert(collection_name="pdf_collection", points=points)



from langgraph.graph import Graph
from langgraph.backends.qdrant import QdrantBackend
from sentence_transformers import SentenceTransformer

# Initialize SentenceTransformer model
model = SentenceTransformer('BAAI/bge-large-en')

# Initialize Qdrant Backend
qdrant_backend = QdrantBackend(
    client=QdrantClient(host="localhost", port=6333),
    collection_name="pdf_collection",
    embedding=model
)

# Initialize the Graph
graph = Graph(backend=qdrant_backend)

