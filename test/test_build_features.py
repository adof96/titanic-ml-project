import pandas as pd
import pytest

from src.features.build_features import (
    agregar_tamano_familia,
    agregar_is_alone,
    agregar_title
)


def test_agregar_tamano_familia_calcula_correctamente():

    # Arrange
    df = pd.DataFrame({
        "SibSp": [1, 0, 3, 0],
        "Parch": [2, 0, 1, 0]
    })

    # Act
    resultado = agregar_tamano_familia(df)

    # Assert
    assert list(resultado["FamilySize"]) == [4, 1, 5, 1]

@pytest.mark.parametrize(
    "family_size, expected",
    [
        (1, 1),
        (2, 0),
        (3, 0),
        (5, 0)
    ]
)
def test_agregar_is_alone(family_size, expected):

    # Arrange
    df = pd.DataFrame({
        "FamilySize": [family_size]
    })

    # Act
    resultado = agregar_is_alone(df)

    # Assert
    assert resultado["IsAlone"].iloc[0] == expected

@pytest.mark.parametrize(
    "name, expected_title",
    [
        ("Smith, Mr. John", "Mr"),
        ("Johnson, Mrs. Mary", "Mrs"),
        ("Brown, Miss. Anna", "Miss"),
        ("Jones, Master. Tim", "Master"),
        ("Doe, Dr. Robert", "Rare")
    ]
)

def test_agregar_title(name, expected_title):

    # Arrange
    df = pd.DataFrame({
        "Name": [name]
    })

    # Act
    resultado = agregar_title(df)

    # Assert
    assert resultado["Title"].iloc[0] == expected_title