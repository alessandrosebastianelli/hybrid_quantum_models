import sys
sys.path += ['.', './hqm/']

from hqm.circuits.angleencoding import BasicEntangledCircuit, StronglyEntangledCircuit
from hqm.regression.hmlp import BasicHybridMLPRegressor, MultiHybridMLPRegressor, MultiHybridMultiMLPRegressor
from hqm.utils.printer import print_circuit

import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np
import torch


# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


n_qubits = 4
n_layers = 2
in_dim   = 48
ou_dim   = 48

dev = qml.device("default.qubit", wires=n_qubits)

qcircuits = [
    BasicEntangledCircuit(n_qubits=n_qubits, n_layers=n_layers, aiframework='torch', dev=dev),
    StronglyEntangledCircuit(n_qubits=n_qubits, n_layers=n_layers, aiframework='torch', dev=dev),
]

model = MultiHybridMultiMLPRegressor(qcircuits, in_dims=[in_dim, n_qubits], ou_dim=ou_dim)



def dataset(n_samples, in_dim, ou_dim):
    X = np.zeros((n_samples, in_dim))
    y = np.zeros((n_samples, ou_dim))
    t = np.linspace(0, 8*np.pi, in_dim + ou_dim)

    for i in range(n_samples):
        a   = np.random.randn(1)
        sig =  a * np.sin(t)
        sig[sig < 0 ] = 0
        X[i, :] = sig[:in_dim]
        y[i, :] = sig[in_dim:]

    return t, X, y


DATASET_SIZE = 400

t, X, y = dataset(DATASET_SIZE, in_dim, ou_dim)
X    = torch.tensor(X, requires_grad=True).float()
y    = torch.tensor(y).float()


opt  = torch.optim.Adam(model.parameters(), lr=0.002)
loss = torch.nn.L1Loss()

BATCH_SIZE = 16
batches = DATASET_SIZE // BATCH_SIZE

data_loader = torch.utils.data.DataLoader(
    list(zip(X, y)), batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)

EPOCHS = 30

#Training Loop
for epoch in range(EPOCHS):
    running_loss = 0
    for xs, ys in data_loader:
        opt.zero_grad()
        loss_evaluated = loss(model(xs), ys)
        loss_evaluated.backward()
        opt.step()
        running_loss += loss_evaluated
    avg_loss = running_loss / batches
    print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))




y_pred = model(X)

fig, axs = plt.subplots(nrows=1, ncols=1)
axs.plot(t[:in_dim], X[0, ...].detach().numpy(), '-*', label='Input')
axs.plot(t[in_dim:], y[0, ...].detach().numpy(), '-*', label='GT')
axs.plot(t[in_dim:], y_pred[0, ...].detach().numpy(), '-*', label='Pred')
axs.legend()

plt.show()