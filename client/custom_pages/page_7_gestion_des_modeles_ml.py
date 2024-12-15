import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL")

def main():
    st.title("Gestion des Modèles ML")
    # Authentification
    with st.sidebar:
        st.subheader("Authentification")
        username = st.text_input("Nom d'utilisateur", key="username")
        password = st.text_input("Mot de passe", type="password", key="password")
        if st.button("Se connecter", key="login_button"):
            try:
                # Requête POST pour s'authentifier
                response = requests.post(
                    f"{API_URL}/auth/token",
                    data={"username": username, "password": password},
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                if response.status_code == 200:
                    token_data = response.json()
                    token = token_data.get("access_token")
                    st.session_state["token"] = token
                    st.session_state["logged_in"] = True
                    st.success("Connexion réussie.")
                else:
                    error_message = response.json().get('detail', 'Erreur d\'authentification.')
                    st.error(f"Erreur : {error_message}")
            except Exception as e:
                st.error(f"Erreur lors de l'authentification : {e}")

    # Vérification du jeton
    token = st.session_state.get("token", None)
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    if not token:
        st.warning("Veuillez vous connecter pour accéder aux fonctionnalités.")
    else:
        # Onglets des fonctionnalités
        tabs = st.tabs(["Liste des Modèles", "Télécharger un Modèle", "Gérer un Modèle"])

        # Onglet 1 : Modèles disponibles
        with tabs[0]:
            st.header("Modèles Disponibles")
            if st.button("Charger la liste des modèles", key="load_models"):
                try:
                    response = requests.get(f"{API_URL}/loopml/available_models", headers=headers)
                    if response.status_code == 200:
                        models = response.json()
                        st.write(models)
                    else:
                        st.error(f"Erreur : {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Erreur lors de la récupération des modèles : {str(e)}")

        # Onglet 2 : Télécharger un modèle
        with tabs[1]:
            st.header("Télécharger un Modèle")
            artifact_name = st.text_input("Nom du modèle à télécharger", key="download_name")
            version = st.text_input("Version (optionnel)", key="download_version")
            if st.button("Télécharger le modèle", key="download_model"):
                try:
                    params = {"artifact_name": artifact_name}
                    if version:
                        params["version"] = version
                    response = requests.get(f"{API_URL}/loopml/download_model_artifact", headers=headers, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        download_path = data.get("download_path")
                        st.success(f"Modèle téléchargé avec succès : {download_path}")
                    else:
                        st.error(f"Erreur : {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Erreur lors du téléchargement : {str(e)}")

        # Onglet 3 : Gestion des modèles
        with tabs[2]:
            st.header("Gestion des Modèles")
            artifact_name = st.text_input("Nom du modèle à supprimer", key="delete_name")
            version = st.text_input("Version (optionnel)", key="delete_version")
            if st.button("Supprimer le modèle", key="delete_model"):
                try:
                    data = {"artifact_name": artifact_name, "version": version} if version else {"artifact_name": artifact_name}
                    response = requests.delete(f"{API_URL}/loopml/delete_model", headers=headers, json=data)
                    if response.status_code == 200:
                        st.success(f"Modèle supprimé avec succès : {artifact_name}")
                    else:
                        st.error(f"Erreur : {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Erreur lors de la suppression : {str(e)}")
