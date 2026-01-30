import numpy as np
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler
import torch
import torch.nn as nn
from qiskit_machine_learning.connectors import TorchConnector


def run_qbm_simulation(X, y):

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    
    feature_map = ZZFeatureMap(feature_dimension=X.shape[1], reps=1)

    sampler = Sampler()

 
    input_params = feature_map.parameters

    qnn = SamplerQNN(
        circuit=feature_map,
        sampler=sampler,
        input_params=input_params,
        weight_params=[]
    )

    model = TorchConnector(qnn)

    class QuantumClassifier(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.qnn = model
            self.fc = nn.Linear(1, len(np.unique(y))) 

        def forward(self, x):
            x = self.qnn(x)
            x = self.fc(x)
            return x

    clf = QuantumClassifier(model)

    return clf, X_scaled
