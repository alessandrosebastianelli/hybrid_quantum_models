import sys
sys.path += ['.', './hqm/']

from hqm.circuits.angleencoding import BasicEntangledCircuit, StronglyEntangledCircuit, RandomCircuit
from hqm.regression.hmlp import BasicHybridMLPRegressor, MultiHybridMLPRegressor, MultiHybridMultiMLPRegressor
from hqm.layers.basiclayer import BasicLayer
from hqm.utils.printer import Printer

import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np
import argparse
import torch

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


def dataset(n_samples : int, in_dim : int, ou_dim : int) -> tuple[np.array, np.array, np.array]:
    '''
        Generate positive sinusoidal signals with random amplitudes.

        Parameters:
        -----------
        - n_samples : int
            number of samples to generate
        - in_din : int
            dimension of the input
        - ou_dim : int
            dimension of the ground truth
        
        Returns:
        --------
        - t : np.array
            array of temporal axis
        - X : np.array
            array containing input samples
        - y: np.array
            array containing ground truth samples
    '''
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test Torch Quantum MLP Regressor')
    # Model's arguments
    parser.add_argument('--dataset_size',           type=int,   default=400)
    parser.add_argument('--n_qubits',               type=int,   default=8)
    parser.add_argument('--n_layers',               type=int,   default=4)
    parser.add_argument('--in_dim',                 type=int,   default=48)
    parser.add_argument('--ou_dim',                 type=int,   default=48)
    parser.add_argument('--batch_size',             type=int,   default=10)
    parser.add_argument('--learning_rate',          type=float, default=0.002)
    parser.add_argument('--epochs',                 type=int,   default=30)
    parser.add_argument('--regressor',              type=str,   default='BasicHybridMLPRegressor')
    parser.add_argument('--circuit',                type=str,   default='BasicEntangledCircuit')
    
    # Reading arguments
    args = parser.parse_args()
    
    
    DATASET_SIZE  = int(args.dataset_size)
    N_QUBITS      = int(args.n_qubits)
    N_LAYERS      = int(args.n_layers)
    IN_DIM        = int(args.in_dim)
    OU_DIM        = int(args.ou_dim)
    BATCH_SIZE    = int(args.batch_size)
    LEARNING_RATE = float(args.learning_rate)
    EPOCHS        = int(args.epochs)
    REGRESSOR     = str(args.regressor)
    CIRCUIT       = str(args.circuit)


    if DATASET_SIZE < 1:   raise Exception('dataset_size must be greater than 1, found {DATASET_SIZE}')
    if BATCH_SIZE < 1:     raise Exception('batch_size must be greater than 1, found {BATCH_SIZE}')
    if LEARNING_RATE <= 0: raise Exception('learning_rate must be greater than 0, found {LEARNING_RATE}')
    if EPOCHS < 1:         raise Exception('epochs must be greater than 0, found {EPOCHS}')
    if REGRESSOR not in ['BasicHybridMLPRegressor', 'MultiHybridMLPRegressor', 'MultiHybridMultiMLPRegressor']:
         raise Exception('regressor can be ''BasicHybridMLPRegressor'', ''MultiHybridMLPRegressor'' or ''MultiHybridMultiMLPRegressor'', found {REGRESSOR}')
    if CIRCUIT not in ['BasicEntangledCircuit', 'StronglyEntangledCircuit', 'RandomCircuit']:
         raise Exception('circuit can be ''BasicEntangledCircuit'', ''StronglyEntangledCircuit'', ''RandomCircuit'', found {CIRCUIT}')

    #=======================================================================
    # Load Dataset
    #=======================================================================
    print('Loading Dataset', '\n')
    t, X, y = dataset(DATASET_SIZE, IN_DIM, OU_DIM)
    X    = torch.tensor(X, requires_grad=True).float()
    y    = torch.tensor(y).float()

    print('Dataset size ', DATASET_SIZE, ' - X shape', X.shape, ' - y shape', y.shape, '\n')
    
    #=======================================================================
    # Inizialize Hybrid Model
    #=======================================================================
    print('Initializing hybrid model', '\n')
    dev = qml.device("default.qubit", wires=N_QUBITS)

    if   CIRCUIT == 'BasicEntangledCircuit':    qcircuit = BasicEntangledCircuit(n_qubits=N_QUBITS, n_layers=N_LAYERS, dev=dev)
    elif CIRCUIT == 'StronglyEntangledCircuit': qcircuit = StronglyEntangledCircuit(n_qubits=N_QUBITS, n_layers=N_LAYERS, dev=dev)
    elif CIRCUIT == 'RandomCircuit':            qcircuit = RandomCircuit(n_qubits=N_QUBITS, n_layers=N_LAYERS, dev=dev)

    if   REGRESSOR == 'BasicHybridMLPRegressor':      model = BasicHybridMLPRegressor(BasicLayer(qcircuit,aiframework='torch'), in_dim=IN_DIM, ou_dim=OU_DIM)
    elif REGRESSOR == 'MultiHybridMLPRegressor':      model = MultiHybridMLPRegressor([BasicLayer(qcircuit,aiframework='torch'),BasicLayer(qcircuit,aiframework='torch')], in_dim=IN_DIM, ou_dim=OU_DIM)
    elif REGRESSOR == 'MultiHybridMultiMLPRegressor': model = MultiHybridMultiMLPRegressor([BasicLayer(qcircuit,aiframework='torch'), BasicLayer(qcircuit,aiframework='torch')], in_dims=[IN_DIM, N_QUBITS], ou_dim=OU_DIM)
    
    Printer.draw_circuit(qcircuit)

    #=======================================================================
    # Train Model
    #=======================================================================
    print('Start training', '\n')
    opt  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss = torch.nn.L1Loss()
    batches = DATASET_SIZE // BATCH_SIZE
    data_loader = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
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

    #=======================================================================
    # Plor results
    #=======================================================================
    print('Plotting results \n')
    y_pred = model(X)
    fig, axs = plt.subplots(nrows=1, ncols=1)
    axs.plot(t[:IN_DIM], X[0, ...].detach().numpy(), '-*', label='Input')
    axs.plot(t[IN_DIM:], y[0, ...].detach().numpy(), '-*', label='GT')
    axs.plot(t[IN_DIM:], y_pred[0, ...].detach().numpy(), '-*', label='Pred')
    axs.legend()
    plt.show()