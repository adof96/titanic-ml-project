import streamlit as st

# ============================
# Configuración de la página
# ============================

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# ============================
# Encabezado
# ============================

st.title("🚢 Titanic Survival Predictor")

st.write(
    """
    Esta aplicación utiliza un modelo de Machine Learning para predecir
    la probabilidad de supervivencia de un pasajero del Titanic.
    """
)