import sys
sys.path += ['.', './circuits/']

from hqm.circuits.circuit import QuantumCircuit
import numpy as np
import torch

class HybridLeNet5(torch.nn.Module):
    '''
        This class implements a quantum hybrid convolutional neural network based on LeNet-5.
        HybridLeNet5 is composed of classical convlutional block and hybrid quantum MLP.
        The size of the network output is defined by ou_dim.
    '''

    def __init__(self, qcircuit : QuantumCircuit, in_shape : tuple, ou_dim : int) -> None:
        '''
            HybridLeNet5 constructor.  

            Parameters:  
            -----------  
            - qcircuit : hqm.circuits.circuit.QuantumCircuit  
                hqm quantum circuit to be stacked between two fully connected layers  
            - in_shape : tuple  
                tuple representing the shape of the input image  
            - ou_dim : int  
                integer representing the output size of the hybrid model  
            
            Returns:  
            --------  
            Nothing, a HybridLeNet5 object will be created.    
        '''
        super().__init__()

        if len(in_shape) != 3: raise Exception(f"The parameter in_shape must be a tuple of three elements, found {in_shape}")
        if ou_dim < 1: raise Exception(f"The parameter ou_dim must be greater than 1, found {ou_dim}")
        
        w, h, c = in_shape
        
        c1 = 6
        self.conv_1    = torch.nn.Conv2d(in_channels=c, out_channels=c1, kernel_size=5, padding=2, stride=1)
        w1 = self.size_flat_features(w, kernel_size=5, padding=2, stride=1)
        h1 = self.size_flat_features(h, kernel_size=5, padding=2, stride=1)
        
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size = (2,2), stride=(2,2))
        w2 = self.size_flat_features(w1, kernel_size=2, padding=0, stride=2)
        h2 = self.size_flat_features(h1, kernel_size=2, padding=0, stride=2)
        
        c2 = 16
        self.conv_2  = torch.nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=5,  stride=1)
        w3 = self.size_flat_features(w2, kernel_size=5, padding=0, stride=1)
        h3 = self.size_flat_features(h2, kernel_size=5, padding=0, stride=1)

        self.max_pool2 = torch.nn.MaxPool2d(kernel_size = (2,2), stride=(2,2))
        w4 = self.size_flat_features(w3, kernel_size=2, padding=0, stride=2)
        h4 = self.size_flat_features(h3, kernel_size=2, padding=0, stride=2)

        

        self.flatten_size = w4 * h4 * c2
        fc_2_size = int(self.flatten_size * 30 / 100)

        self.fc_1    = torch.nn.Linear(self.flatten_size, fc_2_size)
        self.fc_2    = torch.nn.Linear(fc_2_size, qcircuit.n_qubits)
        self.qc_1    = qcircuit.qlayer
        self.fc_3    = torch.nn.Linear(qcircuit.n_qubits, ou_dim)
        self.relu    = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
    
    def size_flat_features(self, s : int, kernel_size : int, padding : int, stride : int) -> int:
        '''
            Get the number of features in a batch of tensor 'x'.  

            Parameters:  
            -----------  
            - s : int  
                integer represeting the size of one axis of the image  
            - kernel_size : int  
                integer represeting the size of the convolutional kernel  
            - padding : int  
                integer represeting the padding size  
            - stride : int  
                integer representing the stride size  
  
            Returns:  
            --------  
            - size : int  
                size after conv2D and Maxpool  
        '''

        size = int(((s - kernel_size + 2 * padding)/stride) + 1)
        return size

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        '''
            Torch forward method  

            Parameters:  
            -----------
            - x : torch.Tensor  
                input for the torch model  

            Returns:  
            --------  
            - out : torch.Tensor  
                output from the torch model  
        '''
        x = self.max_pool1(self.relu(self.conv_1(x)))
        x = self.max_pool2(self.relu(self.conv_2(x)))
        x = x.view(-1, self.flatten_size)
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        x = self.relu(self.qc_1(x))
        x = self.fc_3(x)
        out = self.softmax(x)
        return out