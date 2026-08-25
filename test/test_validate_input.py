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

@pytest.mark.parametrize(
    "campo, valor_invalido, mensaje_esperado",
    [
        ("name", "", "El nombre no puede estar vacío."),
        ("pclass", 4, "La clase del pasajero debe ser 1, 2 o 3."),
        ("sex", "unknown", "El sexo debe ser válido."),
        ("age", 0, "La edad debe ser mayor que 0."),
        ("fare", -1, "El precio del boleto no puede ser negativo."),
        (
            "sibsp",
            -1,
            "El número de hermanos o cónyuges no puede ser negativo."
        ),
        (
            "parch",
            -1,
            "El número de padres o hijos no puede ser negativo."
        )
    ]
)
def test_validar_datos_detecta_datos_invalidos(
    campo,
    valor_invalido,
    mensaje_esperado
):

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

    datos[campo] = valor_invalido

    # Act
    errores = validar_datos(**datos)

    # Assert
    assert mensaje_esperado in errores

@pytest.mark.parametrize(
    "campo, valor, debe_ser_valido",
    [
        ("age", 1, True),
        ("age", 0, False),
        ("fare", 0, True),
        ("fare", -1, False)
    ]
)
def test_limites_de_validacion(campo, valor, debe_ser_valido):

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

    datos[campo] = valor

    # Act
    errores = validar_datos(**datos)

    # Assert
    if debe_ser_valido:
        assert errores == []
    else:
        assert len(errores) > 0

def test_validar_datos_detecta_multiples_errores():

    # Arrange
    datos = {
        "name": "",
        "pclass": 4,
        "sex": "unknown",
        "age": 0,
        "fare": -10,
        "sibsp": -1,
        "parch": -1
    }

    # Act
    errores = validar_datos(**datos)

    # Assert
    assert len(errores) == 7