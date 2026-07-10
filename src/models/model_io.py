from pathlib import Path

import joblib


def guardar_modelo(modelo, ruta: Path):
    """
    Guarda un modelo entrenado en disco.

    Parameters
    ----------
    modelo : object
        Modelo entrenado.

    ruta : Path
        Ruta donde se guardará el modelo.
    """

    ruta.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(modelo, ruta)

    print(f"Modelo guardado en: {ruta}")


def cargar_modelo(ruta: Path):
    """
    Carga un modelo previamente guardado.

    Parameters
    ----------
    ruta : Path
        Ruta del modelo.

    Returns
    -------
    object
        Modelo cargado.
    """

    modelo = joblib.load(ruta)

    print(f"Modelo cargado desde: {ruta}")

    return modelo