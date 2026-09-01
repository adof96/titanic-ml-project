import pytest
from src.models.model_io import guardar_modelo, cargar_modelo


def test_guardar_y_cargar_modelo(tmp_path):

    # Arrange
    modelo_original = {"nombre": "modelo_prueba"}
    ruta = tmp_path / "modelo.pkl"

    # Act
    guardar_modelo(modelo_original, ruta)
    modelo_cargado = cargar_modelo(ruta)

    # Assert
    assert modelo_cargado == modelo_original

def test_guardar_modelo_crea_directorios_si_no_existen(tmp_path):

    # Arrange
    modelo = {"nombre": "modelo_prueba"}

    ruta = (
        tmp_path
        / "carpeta_inexistente"
        / "subcarpeta"
        / "modelo.pkl"
    )

    # Act
    guardar_modelo(modelo, ruta)

    # Assert
    assert ruta.exists()

def test_cargar_modelo_ruta_inexistente(tmp_path):
    # Arrange
    ruta_inexistente = tmp_path / "modelo_inexistente.pkl"

    # Act + Assert
    with pytest.raises(FileNotFoundError):
        cargar_modelo(ruta_inexistente)