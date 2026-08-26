from src.inference.predictor import predecir_pasajero

class ModeloFalso:

    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        return [[0.25, 0.75]]

def test_predecir_pasajero_devuelve_resultado_correcto():

    # Arrange
    modelo = ModeloFalso()

    # Act
    resultado = predecir_pasajero(
        modelo=modelo,
        name="Smith, Mr. John",
        pclass=1,
        sex="male",
        age=30,
        fare=50,
        sibsp=0,
        parch=0
    )

    # Assert
    assert resultado["prediction"] == 1
    assert resultado["prob_not_survival"] == 0.25
    assert resultado["prob_survival"] == 0.75