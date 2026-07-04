import joblib


def guardar_modelo(modelo, ruta):
    """
    Guarda un modelo entrenado en disco.

    Parameters
    ----------
    modelo : object
        Modelo entrenado.

    ruta : str
        Ruta donde se guardará el modelo.
    """

    joblib.dump(modelo, ruta)

    print(f"Modelo guardado en: {ruta}")


def cargar_modelo(ruta):
    """
    Carga un modelo previamente guardado.

    Parameters
    ----------
    ruta : str
        Ruta del modelo.

    Returns
    -------
    object
        Modelo cargado.
    """

    modelo = joblib.load(ruta)

    print(f"Modelo cargado desde: {ruta}")

    return modelo