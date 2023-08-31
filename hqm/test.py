from circuits.angle_encoding import BasicEntangledCircuit, StronglyEntangledCircuit
from classifiers.HybridMLP import HybridMLP
from utils.printer import print_circuit

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np
import torch

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

X, y  = make_moons(n_samples=200, noise=0.1)
y_    = torch.unsqueeze(torch.tensor(y), 1)  # used for one-hot encoded labels
y_hot = torch.scatter(torch.zeros((200, 2)), 1, y_, 1)

n_qubits = 2
n_layers = 6

qcircuits = [
    BasicEntangledCircuit(n_qubits=n_qubits, n_layers=n_layers, framework='torch'),
    StronglyEntangledCircuit(n_qubits=n_qubits, n_layers=n_layers, framework='torch')
]

print_circuit(qcircuits[1])

model = HybridMLP(qcircuits[1])







opt   = torch.optim.SGD(model.parameters(), lr=0.2)
loss  = torch.nn.L1Loss()

X     = torch.tensor(X, requires_grad=True).float()
y_hot = y_hot.float()

batch_size = 5
batches = 200 // batch_size

data_loader = torch.utils.data.DataLoader(
    list(zip(X, y_hot)), batch_size=5, shuffle=True, drop_last=True
)

epochs = 15

for epoch in range(epochs):

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
predictions = torch.argmax(y_pred, axis=1).detach().numpy()

correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, y)]
accuracy = sum(correct) / len(correct)
print(f"Accuracy: {accuracy * 100}%")
