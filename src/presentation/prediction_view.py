import streamlit as st

def mostrar_resultado(resultado):
    """
    Renderiza los componentes visuales de Streamlit para mostrar
    los resultados de la predicción de supervivencia.
    """
    # Mostrar el resultado principal
    if resultado["prediction"] == 1:
        st.success(
            "🎉 El modelo predice que el pasajero probablemente sobreviviría."
        )
    else:
        st.error(
            "⚠️ El modelo predice que el pasajero probablemente no sobreviviría."
        )

    # Mostrar las probabilidades
    st.subheader("Resultados de la predicción")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="✅ Probabilidad de supervivencia",
            value=f"{resultado['prob_survival']:.2%}"
        )

    with col2:
        st.metric(
            label="❌ Probabilidad de no supervivencia",
            value=f"{resultado['prob_not_survival']:.2%}"
        )

    # Barra de progreso
    st.write("Probabilidad de supervivencia")
    st.progress(resultado["prob_survival"])

    # Explicación para el usuario
    st.caption(
        "La probabilidad representa la confianza del modelo en su predicción. "
        "No garantiza que el evento ocurra, sino la estimación realizada por el modelo "
        "a partir de los patrones aprendidos durante el entrenamiento."
    )