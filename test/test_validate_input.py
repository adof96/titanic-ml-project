import pytest
from src.validation.validate_input import validar_datos

def test_validar_datos_con_datos_validos():

    # Arrange
    datos = {
        "name": "Smith, Mr. John",
        "pclass": 1,
        "sex": "male",
        "age": 30,
        "fare": 50,
        "sibsp": 0,
        "parch": 0
    }

    # Act
    errores = validar_datos(**datos)

    # Assert
    assert errores == []