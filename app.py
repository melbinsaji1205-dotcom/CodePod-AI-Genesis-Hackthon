# app.py
import streamlit as st
from matching import get_recommendations, embed_text
from qdrant_client import QdrantClient

client = QdrantClient(
    url=""https://4a8c79c1-1d51-435f-92ea-4fbb28af3f11.us-west-1-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.JLpMaR2JIwITPOAgRUnFBT3rjciZoZrZ91YgEOgs9ro"                              # your API key
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



