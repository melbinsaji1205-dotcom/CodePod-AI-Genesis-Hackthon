# matching.py
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import minmax_scale

client = QdrantClient(
    url="https://0b81db2c-c6b1-4ad8-85ae-41330568295f.europe-west3-0.gcp.cloud.qdrant.io:6333",  # your Qdrant Cloud URL
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.P9qf7bJaV2hpzpkgIUlusSFVyJLUz3jHrXDI8X6SMgU"                              # your API key
)
COLLECTION = "db"
text_model = SentenceTransformer("all-mpnet-base-v2")

def embed_text(text):
    return text_model.encode(text, normalize_embeddings=True)

def jaccard(a, b):
    sa = set([x.lower().strip() for x in a])
    sb = set([x.lower().strip() for x in b])
    if not sa and not sb: return 0.0
    return len(sa & sb) / len(sa | sb)


ACCOMMODATION_MATCH = {
    # candidate disability -> jobs they excel in (example)
    "visual_impairment": {"phone-based", "data-entry", "customer-support"},
    "hearing_impairment": {"data-entry", "software-testing", "backend-development"},
    "mobility_impairment": {"remote-work", "desk-jobs", "admin"},
    "intellectual_disability": {"supported-employment", "assembly", "repetitive-tasks"}
}

def compute_accessibility_fit(candidate_payload, job_requirements_tags):
    # returns 0..1
    d = candidate_payload.get("disability_type","")
    allowed = ACCOMMODATION_MATCH.get(d, set())
    overlap = len(allowed & set(job_requirements_tags))
    return min(1.0, overlap / max(1, len(job_requirements_tags)))

def get_recommendations(job_text, requested_skills, top_k=50, inclusion_policy_score=0.5):
    job_emb = embed_text(job_text).tolist()
    hits = client.search(collection_name=COLLECTION, query_vector=job_emb, limit=top_k*3)

    results = []
    match_scores = [h.score for h in hits] 
    match_scores_norm = minmax_scale(match_scores)
    for i, h in enumerate(hits):
        payload = h.payload
        skill_overlap = jaccard(payload.get("skills", []), requested_skills)
        accessibility_fit = compute_accessibility_fit(payload, set(requested_skills))
        normalized_match = float(match_scores_norm[i])
        inclusion_score = 0.55 * normalized_match + 0.25 * skill_overlap + 0.20 * accessibility_fit
        results.append({
            "candidate_id": payload.get("candidate_id"),
            "name": payload.get("name"),
            "match_score": normalized_match,
            "skill_overlap": skill_overlap,
            "accessibility_fit": accessibility_fit,
            "inclusion_impact_score": inclusion_score,
            "payload": payload
        })
    # Sort descending by inclusion impact score
    results = sorted(results, key=lambda x: x["inclusion_impact_score"], reverse=True)
    return results



