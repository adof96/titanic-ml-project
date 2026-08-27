import pytest
from src.inference.predictor import predecir_pasajero
from src.config import FEATURES
class ModeloFalso:

    def __init__(self, prediccion, probabilidades):
        self.prediccion = prediccion
        self.probabilidades = probabilidades
        self.datos_recibidos = None

    def predict(self, X):
        self.datos_recibidos = X
        return [self.prediccion]

    def predict_proba(self, X):
        return [self.probabilidades]

@pytest.mark.parametrize(
    "prediccion, probabilidades, prob_not_survival, prob_survival",
    [
        (1, [0.25, 0.75], 0.25, 0.75),
        (0, [0.80, 0.20], 0.80, 0.20),
    ]
)
def test_predecir_pasajero(
    prediccion,
    probabilidades,
    prob_not_survival,
    prob_survival
):

    # Arrange
    modelo = ModeloFalso(prediccion, probabilidades)

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
    assert resultado["prediction"] == prediccion
    assert resultado["prob_not_survival"] == prob_not_survival
    assert resultado["prob_survival"] == prob_survival
    assert list(modelo.datos_recibidos.columns) == FEATURES