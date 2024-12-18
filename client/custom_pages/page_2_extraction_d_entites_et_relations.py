import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL")

def main():
    st.title("Extraction d'Entités & Relation")
    input_text = st.text_area("Entrez un texte pour extraction")
    if st.button("Extraire"):
        with st.spinner("Extraction en cours..."):
            response = requests.post(
                f"{API_URL}/extract-entities-relationships", json={"text": input_text}
            )
            if response.status_code == 200:
                data = response.json()
                st.subheader("Entités")
                st.json(data.get("entities", {}))
                st.subheader("Relations")
                st.json(data.get("relationships", {}))
            else:
                st.error(f"Erreur : {response.status_code} - {response.text}")
