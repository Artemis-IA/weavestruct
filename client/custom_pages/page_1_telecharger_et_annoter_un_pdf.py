# client/pages/1_Télécharger_et_Annoter_un_PDF.py
import streamlit as st
import requests
import os
from PyPDF2 import PdfReader

# Load API URL
API_URL = os.getenv("API_URL")

# Helper function to upload a PDF to the API
def upload_pdf(file, params):
    files = {"files": (file.name, file.getvalue())}
    response = requests.post(
        f"{API_URL}/documents/upload/",
        files=files,
        params=params
    )
    return response

# Helper function to process a folder
def upload_folder(directory_path, params):
    response = requests.post(
        f"{API_URL}/documents/upload_path/",
        data={"directory_path": directory_path},
        params=params
    )
    return response

# Helper function to process indexing
def index_document(file, params):
    files = {"file": (file.name, file.getvalue())} if file else None
    response = requests.post(
        f"{API_URL}/documents/index_document/",
        files=files,
        params=params
    )
    return response

# Main function
def main():
    st.title("📄 Télécharger et Annoter un PDF")
    st.sidebar.title("Options de traitement")
    
    # Choose processing method
    mode = st.sidebar.radio("Mode de traitement :", ["Télécharger un PDF", "Indexer un dossier", "Indexer un document"])

    # Common options for all modes
    use_ocr = st.sidebar.checkbox("Utiliser OCR", value=False)
    export_figures = st.sidebar.checkbox("Exporter les figures", value=True)
    export_tables = st.sidebar.checkbox("Exporter les tableaux", value=True)
    enrich_figures = st.sidebar.checkbox("Enrichir les figures", value=False)
    export_formats = st.sidebar.multiselect(
        "Formats d'exportation :", 
        ["json", "yaml", "md", "text"], 
        default=["json"]
    )
    params = {
        "use_ocr": use_ocr,
        "export_figures": export_figures,
        "export_tables": export_tables,
        "enrich_figures": enrich_figures,
        "export_formats": export_formats,
    }

    # Mode: Upload a single PDF
    if mode == "Télécharger un PDF":
        uploaded_pdf = st.file_uploader("Télécharger un PDF", type=["pdf"])
        if uploaded_pdf:
            with st.spinner("Traitement du PDF..."):
                response = upload_pdf(uploaded_pdf, params)
                if response.status_code == 200:
                    st.success("PDF traité avec succès !")
                    data = response.json()
                    st.subheader("Annotations et Résultats")
                    st.json(data)
                    st.subheader("Contenu Original du PDF")
                    pdf_reader = PdfReader(uploaded_pdf)
                    for i, page in enumerate(pdf_reader.pages):
                        st.markdown(f"### Page {i+1}")
                        st.text(page.extract_text())
                else:
                    st.error(f"Erreur lors du traitement : {response.status_code} - {response.text}")

    # Mode: Process a folder
    elif mode == "Indexer un dossier":
        directory_path = st.text_input("Chemin du dossier local contenant les fichiers PDF")
        if st.button("Indexer le dossier"):
            if not directory_path:
                st.warning("Veuillez fournir un chemin valide.")
            else:
                with st.spinner("Indexation en cours..."):
                    response = upload_folder(directory_path, params)
                    if response.status_code == 200:
                        st.success(f"Dossier indexé avec succès : {response.json()}")
                    else:
                        st.error(f"Erreur lors de l'indexation : {response.status_code} - {response.text}")

    # Mode: Index a single document
    elif mode == "Indexer un document":
        uploaded_file = st.file_uploader("Télécharger un document à indexer", type=["pdf", "docx"])
        if st.button("Indexer le document") and uploaded_file:
            with st.spinner("Indexation en cours..."):
                response = index_document(uploaded_file, params)
                if response.status_code == 200:
                    st.success("Document indexé avec succès !")
                    st.json(response.json())
                else:
                    st.error(f"Erreur lors de l'indexation : {response.status_code} - {response.text}")

if __name__ == "__main__":
    main()
