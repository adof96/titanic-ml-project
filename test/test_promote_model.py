from src.models.model_io import guardar_modelo, cargar_modelo
from src.models.promote_model import promover_modelo


def test_promover_modelo(tmp_path):

    # Arrange
    modelo_original = {"nombre": "modelo_validado"}

    ruta_origen = tmp_path / "validado" / "modelo.pkl"
    ruta_destino = tmp_path / "produccion" / "modelo.pkl"

    guardar_modelo(modelo_original, ruta_origen)

    # Act
    promover_modelo(ruta_origen, ruta_destino)

    # Assert
    modelo_promovido = cargar_modelo(ruta_destino)

    assert modelo_promovido == modelo_original