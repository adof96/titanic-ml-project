import pandas as pd

from src.features.build_features import agregar_tamano_familia


def test_agregar_tamano_familia_calcula_correctamente():

    # Arrange
    df = pd.DataFrame({
        "SibSp": [1, 0, 3],
        "Parch": [2, 0, 1]
    })

    # Act
    resultado = agregar_tamano_familia(df)

    # Assert
    assert list(resultado["FamilySize"]) == [4, 1, 5]