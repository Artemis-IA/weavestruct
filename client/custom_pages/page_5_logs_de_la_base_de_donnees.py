import streamlit as st
import requests
import pandas as pd
import os
API_URL = os.getenv("API_URL")

def fetch_logs():
    response = requests.get(f"{API_URL}/logs")
    return response.json() if response.status_code == 200 else []

def main():
    st.title("Logs de la Base de Données")
    logs = fetch_logs()
    if logs:
        df_logs = pd.DataFrame(logs)
        st.dataframe(df_logs)
    else:
        st.warning("Aucun log disponible.")
