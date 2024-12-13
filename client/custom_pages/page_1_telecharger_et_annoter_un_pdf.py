# client/pages/1_Télécharger_et_Annoter_un_PDF.py
import streamlit as st
import requests
from PyPDF2 import PdfReader
import os


def upload_pdf(file):
    files = {"files": (file.name, file.getvalue())}
    API_URL = os.getenv("API_URL")
    response = requests.post(f"{API_URL}/documents/upload/", files=files)
    return response


def main():
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
