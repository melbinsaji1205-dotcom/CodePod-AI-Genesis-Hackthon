import os
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np
import streamlit as st

QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
COLLECTION_NAME = "db"

# Embedding model
text_model = SentenceTransformer("all-mpnet-base-v2")  # 768 dims

# Qdrant client
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) if QDRANT_URL else QdrantClient()

VECTOR_SIZE = 768 

# Create collection 
try:
    client.get_collection(COLLECTION_NAME)
except Exception:
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vector_config=rest.Vectorparams(size-VECTOR_SIZE, distance=rest.Distance.COSINE)
        shard_number=1
    )
    

# Load CSV
df = pd.read_csv("differently_abled_candidates_uae.csv")

def candidate_text_to_embed(row):
    parts = []
    parts.append(str(row.get("skills_qualifications", "")))
    parts.append(str(row.get("trainings_acquired", "")))
    parts.append(str(row.get("area_of_interest", "")))
    parts.append(str(row.get("current_status", "")))
    parts.append(f"aptitude_score:{row.get('aptitude_score_percent', '')}")
    text = " ||| ".join([p for p in parts if p and str(p).strip() != "nan"])
    return text

BATCH = 64
points = []
for idx, row in df.iterrows():
    text_blob = candidate_text_to_embed(row)
    emb = text_model.encode(text_blob, normalize_embeddings=True)
    candidate_id = str(row['id'])
    metadata = {
        "candidate_id": candidate_id,
        "name": row.get("name", ""),
        "disability_type": row.get("disability_type", ""),
        "disability_subcategory": row.get("disability_subcategory", ""),
        "skills": [s.strip() for s in str(row.get("skills_qualifications","")).split(",") if s.strip()],
        "area_of_interest": row.get("area_of_interest", ""),
        "aptitude_score_percent": float(row.get("aptitude_score_percent", 0) or 0),
        "trainings_acquired": [t.strip() for t in str(row.get("trainings_acquired","")).split(",") if t.strip()],
        "location": row.get("nationality", ""),
        "consent_to_share_contact": False 
    }
    points.append(rest.PointStruct(id=int(candidate_id), vector=emb.tolist(), payload=metadata))
    if len(points) >= BATCH:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        points = []
if points:
    client.upsert(collection_name=COLLECTION_NAME, points=points)

print("Ingestion complete.")












