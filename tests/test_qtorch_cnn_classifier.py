import sys
sys.path += ['.', './hqm/']

from hqm.circuits.angleencoding import BasicEntangledCircuit
from hqm.classification.hcnn import HybridLeNet5
from hqm.layers.basiclayer import BasicLayer
from hqm.utils.printer import Printer

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pennylane as qml
import numpy as np
import argparse
import torch
import glob
import os

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


def dataset(path : str, channelfirst : bool) -> tuple[np.array, np.array]:
    '''
        Load EuroSAT dataset.

        Parameters:
        -----------
        - path : str
            root path of the dataset
        - channelfirst : bool
            if true the channel axis will be moved
        Returns:
        --------
        - X : np.array
            array containing input samples
        - y: np.array
            array containing ground truth samples
    '''

    imgs = glob.glob(os.path.join(path, '*', '*'))
    np.random.shuffle(imgs)
    X = np.zeros((len(imgs), 64, 64, 3))
    y = np.zeros((len(imgs), 10))
    classes = {'AnnualCrop':0, 'Forest':1, 'HerbaceousVegetation':2, 'Highway':3, 'Industrial':4, 'Pasture':5, 'PermanentCrop':6, 'Residential':7, 'River':8, 'SeaLake':9}
    

    for i, img in enumerate(imgs):
        
        X[i, ...] = plt.imread(img)/255.0
        
        for c, ic in classes.items():
            if c in img:
                y[i, ic] = 1
                break
    
    if channelfirst:
        X = np.moveaxis(X, -1, 1)

    return X, y


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test Torch Quantum CNN Classifier')
    # Model's arguments
    parser.add_argument('--n_qubits',               type=int,   default=4)
    parser.add_argument('--n_layers',               type=int,   default=2)
    parser.add_argument('--batch_size',             type=int,   default=10)
    parser.add_argument('--learning_rate',          type=float, default=0.002)
    parser.add_argument('--epochs',                 type=int,   default=30)
    
    # Reading arguments
    args = parser.parse_args()
    
    
    N_QUBITS      = int(args.n_qubits)
    N_LAYERS      = int(args.n_layers)
    BATCH_SIZE    = int(args.batch_size)
    LEARNING_RATE = float(args.learning_rate)
    EPOCHS        = int(args.epochs)


    if N_QUBITS < 1:       raise Exception('n_qubits must be greater than 1, found {N_QUBITS}')
    if N_LAYERS < 1:       raise Exception('n_layers must be greater than 1, found {N_LAYERS}')
    if BATCH_SIZE < 1:     raise Exception('batch_size must be greater than 1, found {BATCH_SIZE}')
    if LEARNING_RATE <= 0: raise Exception('learning_rate must be greater than 0, found {LEARNING_RATE}')
    if EPOCHS < 1:         raise Exception('epochs must be greater than 0, found {EPOCHS}')

    #=======================================================================
    # Load Dataset
    #=======================================================================
    print('Loading Dataset', '\n')
    X, y = dataset('datasets/EuroSAT', channelfirst=True)

    X    = torch.tensor(X, requires_grad=True).float()
    y    = torch.tensor(y).float()

    print('Dataset size', ' - X shape', X.shape, ' - y shape', y.shape, '\n')
    
    #=======================================================================
    # Inizialize Hybrid Model
    #=======================================================================
    print('Initializing hybrid model', '\n')
    dev = qml.device("default.qubit", wires=N_QUBITS)

    qcircuit = BasicEntangledCircuit(n_qubits=N_QUBITS, n_layers=N_LAYERS, dev=dev)
    model = HybridLeNet5(qcircuit=BasicLayer(qcircuit, aiframework='torch'), in_shape=(64,64,3), ou_dim=10)
    
    
    Printer.draw_circuit(qcircuit)

    #=======================================================================
    # Train Model
    #=======================================================================
    print('Start training', '\n')
    opt  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss = torch.nn.L1Loss()
    batches = X.shape[0] // BATCH_SIZE
    data_loader = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    #Training Loop
    for epoch in range(EPOCHS):
        running_loss = 0
        for xs, ys in tqdm(data_loader):
            opt.zero_grad()
            loss_evaluated = loss(model(xs), ys)
            loss_evaluated.backward()
            opt.step()
            running_loss += loss_evaluated
        avg_loss = running_loss / batches
        print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))