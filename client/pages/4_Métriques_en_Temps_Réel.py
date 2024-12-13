import streamlit as st
import requests
import pandas as pd
import plotly.express as px

PROMETHEUS_URL = "http://localhost:9090/metrics"

def fetch_metrics():
    response = requests.get(PROMETHEUS_URL)
    return response.text if response.status_code == 200 else None

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
