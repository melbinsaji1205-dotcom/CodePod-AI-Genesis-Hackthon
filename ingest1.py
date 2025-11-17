import pandas as pd
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Load CSV
df = pd.read_csv("differently_abled_candidates_uae.csv").fillna("")

# Initialize embeddings
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Convert dataframe rows to Documents
documents = []
for idx, row in df.iterrows():
    text = " ".join([str(value) for value in row.values])
    metadata = row.to_dict()
    documents.append(page_content=text, metadata=metadata)

print("Embedding model loaded, converting documents...")

# Initialize Qdrant client
client = QdrantClient(url="http://localhost:6333")
client.recreate_collection(
    collection_name="differently_abled",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)

# Create vector store in Qdrant
vector_store = Qdrant.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="differently_abled",
    url="http://localhost:6333",
    prefer_grpc=False
)

print("Qdrant collection created and documents uploaded successfully!")
