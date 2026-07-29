# ============================
# Importaciones
# ============================

import streamlit as st

from src.models.model_io import cargar_modelo
from src.config import PRODUCTION_MODEL_PATH

# ============================
# Configuración
# ============================

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# ============================
# Recursos
# ============================

@st.cache_resource
def cargar_modelo_streamlit():
    """
    Carga el modelo entrenado una única vez.
    """
    return cargar_modelo(PRODUCTION_MODEL_PATH)


modelo = cargar_modelo_streamlit()

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

st.success("✅ Modelo cargado correctamente.")