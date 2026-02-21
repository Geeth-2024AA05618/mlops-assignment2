import numpy as np
from api.app import predict_from_array

class DummyModel:
    def predict(self, x):
        return np.array([[0.7]])

def test_predict_from_array():
    dummy_model = DummyModel()
    dummy_input = np.zeros((1, 224, 224, 3))

    prediction = predict_from_array(dummy_model, dummy_input)

    assert prediction == 0.7