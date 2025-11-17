# app.py
import streamlit as st
from matching import get_recommendations, embed_text
from qdrant_client import QdrantClient

client = QdrantClient(
    url="https://0b81db2c-c6b1-4ad8-85ae-41330568295f.europe-west3-0.gcp.cloud.qdrant.io:6333",  # your Qdrant Cloud URL
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.P9qf7bJaV2hpzpkgIUlusSFVyJLUz3jHrXDI8X6SMgU"                              # your API key
)

st.title("CodePod")

job_text = st.text_area("Enter job description / requirements", height=200)
skills_input = st.text_input("Requested skills (comma-separated)")
top_k = st.slider("How many recommendations?", 5, 50, 10)

if st.button("Find candidates"):
    skills = [s.strip() for s in skills_input.split(",") if s.strip()]
    recs = get_recommendations(job_text, skills, top_k=top_k)
    for r in recs[:top_k]:
        payload = r["payload"]
        st.subheader(f"{payload.get('name')}  — Score: {r['inclusion_impact_score']:.3f}")
        st.write("Skills:", ", ".join(payload.get("skills", [])))
        st.write("Disability:", payload.get("disability_type"))
        st.write("Aptitude %:", payload.get("aptitude_score_percent"))
        if payload.get("consent_to_share_contact"):
            st.write("Contact:", payload.get("phone", "N/A"), payload.get("email", "N/A"))
        else:
            st.write("Contact: (not shared)")
        st.write("---")


