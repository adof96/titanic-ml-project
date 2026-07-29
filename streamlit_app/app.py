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
# Encabezado
# ============================

st.title("Prueba")

st.write(
    """
    Esta aplicación utiliza un modelo de Machine Learning para predecir
    la probabilidad de supervivencia de un pasajero del Titanic.
    """
)
# ============================
# Recursos
# ============================

@st.cache_resource
def cargar_modelo_streamlit():
    """
    Carga el modelo entrenado una única vez y lo reutiliza
    durante toda la ejecución de la aplicación.
    """
    return cargar_modelo(PRODUCTION_MODEL_PATH)


# ============================
# Interfaz
# ============================

# ============================
# Resultados
# ============================