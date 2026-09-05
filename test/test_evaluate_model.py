from src.models.evaluate_model import calcular_metricas


def test_calcular_metricas_matriz_confusion():

    # Arrange
    y_test = [0, 0, 1, 1]
    predictions = [0, 1, 1, 1]

    # Act
    metricas = calcular_metricas(y_test, predictions)

    # Assert
    expected_matrix = [
        [1, 1],
        [0, 2]
    ]

    assert metricas["confusion_matrix"].tolist() == expected_matrix

def test_calcular_metricas_classification_report():

    # Arrange
    y_test = [0, 0, 1, 1]
    predictions = [0, 1, 1, 1]

    # Act
    metricas = calcular_metricas(y_test, predictions)

    # Assert
    assert metricas["classification_report"]["1"]["recall"] == 1.0