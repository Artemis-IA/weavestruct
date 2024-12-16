import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL")

def fetch_metrics():
    response = requests.get(PROMETHEUS_URL)
    return response.text if response.status_code == 200 else None

def main():
    st.title("Métriques en Temps Réel")
    with st.spinner("Récupération des métriques..."):
        metrics = fetch_metrics()
        if metrics:
            lines = metrics.splitlines()
            data = [{"metric": line.split()[0], "value": line.split()[1]} for line in lines if line]
            df = pd.DataFrame(data)
            st.dataframe(df)
            fig = px.line(df, x="metric", y="value", title="Métriques")
            st.plotly_chart(fig)
        else:
            st.warning("Aucune métrique disponible.")
