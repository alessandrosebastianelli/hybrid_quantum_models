import sys
sys.path += ['.', './layers/', './utils/']

from hqm.layers.basiclayer import BasicLayer
from hqm.utils.sizes import size_conv_layer

import torch

class HybridLeNet5(torch.nn.Module):
    '''
        This class implements a quantum hybrid convolutional neural network based on LeNet-5.
        HybridLeNet5 is composed of classical convlutional block and hybrid quantum MLP.
        The size of the network output is defined by ou_dim.
    '''

    def __init__(self, qlayer : BasicLayer, in_shape : tuple, ou_dim : int) -> None:
        '''
            HybridLeNet5 constructor.  

            Parameters:  
            -----------  
            - qlayer : hqm.layers.basilayer.BasicLayer  
                hqm quantum layer to be stacked between two fully connected layers  
            - in_shape : tuple  
                tuple representing the shape of the input image (channels, widht, height)  
            - ou_dim : int  
                integer representing the output size of the hybrid model  
            
            Returns:  
            --------  
            Nothing, a HybridLeNet5 object will be created.    
        '''

        super().__init__()

        if len(in_shape) != 3: raise Exception(f"The parameter in_shape must be a tuple of four elements (channels, widht, height), found {in_shape}")
        if ou_dim < 1: raise Exception(f"The parameter ou_dim must be greater than 1, found {ou_dim}")
        
        c, w, h = in_shape
        
        c1 = 6
        self.conv_1    = torch.nn.Conv2d(in_channels=c, out_channels=c1, kernel_size=5, padding=2, stride=1)
        w1 = size_conv_layer(w, kernel_size=5, padding=2, stride=1)
        h1 = size_conv_layer(h, kernel_size=5, padding=2, stride=1)
        
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size = (2,2), stride=(2,2))
        w2 = size_conv_layer(w1, kernel_size=2, padding=0, stride=2)
        h2 = size_conv_layer(h1, kernel_size=2, padding=0, stride=2)
        
        c2 = 16
        self.conv_2  = torch.nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=5,  stride=1)
        w3 = size_conv_layer(w2, kernel_size=5, padding=0, stride=1)
        h3 = size_conv_layer(h2, kernel_size=5, padding=0, stride=1)

        self.max_pool2 = torch.nn.MaxPool2d(kernel_size = (2,2), stride=(2,2))
        w4 = size_conv_layer(w3, kernel_size=2, padding=0, stride=2)
        h4 = size_conv_layer(h3, kernel_size=2, padding=0, stride=2)

        self.flatten_size = w4 * h4 * c2
        fc_2_size = int(self.flatten_size * 30 / 100)

        self.fc_1    = torch.nn.Linear(self.flatten_size, fc_2_size)
        self.fc_2    = torch.nn.Linear(fc_2_size, qlayer.n_qubits)
        self.qc_1    = qlayer.qlayer
        self.fc_3    = torch.nn.Linear(qlayer.n_qubits, ou_dim)
        self.relu    = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
    
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