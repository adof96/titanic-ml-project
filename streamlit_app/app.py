# ============================
# Importaciones
# ============================

import streamlit as st

from src.models.model_io import cargar_modelo
from src.config import PRODUCTION_MODEL_PATH
from src.inference.predictor import predecir_pasajero
from src.presentation.prediction_view import mostrar_resultado
# ============================
# Configuracións
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

# ============================
# Formulario
# ============================

st.header("Información del pasajero")

name = st.text_input(
    "Nombre",
    placeholder="Ej. John Smith"
)

pclass = st.selectbox(
    "Clase del pasajero",
    options=[1, 2, 3],
    format_func=lambda x: {
        1: "🥇 Primera clase",
        2: "🥈 Segunda clase",
        3: "🥉 Tercera clase"
    }[x]
)

# Explicación de la clase seleccionada

if pclass == 1:
    st.info(
        "🏛️ **Primera clase**\n\n"
        "La opción más exclusiva del Titanic. "
        "Podría compararse con viajar hoy en una experiencia premium "
        "o clase ejecutiva, con acceso a los mejores camarotes y servicios."
    )

elif pclass == 2:
    st.info(
        "💼 **Segunda clase**\n\n"
        "Una opción cómoda para profesionales y familias de clase media. "
        "Ofrecía buenas instalaciones, aunque con menos lujos que la primera clase."
    )

else:
    st.info(
        "🧳 **Tercera clase**\n\n"
        "La alternativa más económica. "
        "Era utilizada principalmente por inmigrantes y trabajadores que "
        "viajaban en busca de nuevas oportunidades."
    )

sex = st.selectbox(
    "Sexo",
    options=["male", "female"],
    format_func=lambda x: {
        "male": "Hombre",
        "female": "Mujer"
    }[x]
)

age = st.number_input(
    "Edad",
    min_value=0.0,
    value=30.0,
    step=1.0
)

fare = st.number_input(
    "Precio del boleto (£)",
    min_value=0.0,
    value=30.0,
    step=1.0,
    help=(
        "Costo del boleto pagado por el pasajero para viajar en el Titanic. "
        "El valor está expresado en libras esterlinas (£) de la época."
    )
)

sibsp = st.number_input(
    "Hermanos o cónyuge a bordo",
    min_value=0,
    value=0,
    step=1,
    help="Número de hermanos(as) o cónyuge que viajaban junto al pasajero."
)

parch = st.number_input(
    "Padres o hijos a bordo",
    min_value=0,
    value=0,
    step=1,
    help="Número de padres o hijos que viajaban junto al pasajero."
)

# ... campos del formulario ...

if st.button("🔍 Realizar predicción"):

    try:

        with st.spinner("Realizando predicción..."):

            resultado = predecir_pasajero(
                name=name,
                pclass=pclass,
                sex=sex,
                age=age,
                fare=fare,
                sibsp=sibsp,
                parch=parch,
                modelo=modelo
            )

        # Invocación de la vista aislada para mostrar los datos procesados
        mostrar_resultado(resultado)

    except Exception as e:

        st.error(
            "Ha ocurrido un error inesperado durante la predicción."
        )

        st.exception(e)

