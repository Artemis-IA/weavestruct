import streamlit as st
import requests
from PyPDF2 import PdfReader

API_URL = "http://localhost:8008/"

def upload_pdf(file):
    files = {"files": (file.name, file.getvalue())}
    response = requests.post(f"{API_URL}/documents/upload/", files=files)
    return response

st.title("Télécharger et Annoter un PDF")
uploaded_pdf = st.file_uploader("Télécharger un PDF", type=["pdf"])
if uploaded_pdf:
    with st.spinner("Traitement du PDF..."):
        response = upload_pdf(uploaded_pdf)
        if response.status_code == 200:
            st.success("PDF traité avec succès !")
            data = response.json()
            st.subheader("Annotations")
            st.json(data)
            st.subheader("Contenu Original")
            pdf_reader = PdfReader(uploaded_pdf)
            for i, page in enumerate(pdf_reader.pages):
                st.markdown(f"### Page {i+1}")
                st.text(page.extract_text())
        else:
            st.error(f"Erreur lors du traitement : {response.status_code} - {response.text}")
