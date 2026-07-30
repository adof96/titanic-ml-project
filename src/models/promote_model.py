from src.models.model_io import cargar_modelo, guardar_modelo


def promover_modelo(ruta_origen, ruta_destino):
    """
    Promueve un modelo validado a modelo de producción.

    Parameters
    ----------
    ruta_origen : Path
        Ruta del modelo validado.

    ruta_destino : Path
        Ruta donde se guardará el modelo de producción.
    """

    modelo = cargar_modelo(ruta_origen)

    guardar_modelo(modelo, ruta_destino)

    print("Modelo promovido correctamente.")