# app.py
import streamlit as st
from matching import get_recommendations, embed_text
from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "db"
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







