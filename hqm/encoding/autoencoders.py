import sys
sys.path += ['.', './layers/', '/circuits/', './utils/']

from hqm.layers.quanvolution import Quanvolution2D
from hqm.circuits.circuit import QuantumCircuit
from hqm.layers.basiclayer import BasicLayer
from hqm.utils.sizes import size_conv_layer

import torch


class QuanvolutionAutoencoder(torch.nn.Module): 
    '''
        Hybrdid convolutional autoencoder, the encoder is composed of quanvolution layers, the decoder is composed of classical layers.
    '''

    def __init__(self, qcircuits : list[QuantumCircuit], in_shape : tuple[int], filters : list[int], kernelsizes : list[int], strides : list[int]) -> None:
        '''
            QuanvolutionAutoencoder constructor.  

            Parameters:  
            -----------
            - qcircuits : list  
                list of QuantumCircuit objects
            - in_shape : tuple 
                tuple represeting the image shape, (width, height, channels)
            - filters : list
                list containing the number of filters for each quanvolution layer
            - kernelsizes: list
                list containing the kernelsize for each quanvolution layer
            - strides: list
                list containin the stride for each quanvolution layer  
            
            Returns:  
            --------  
            Nothing, a QuanvolutionAutoencoder object will be created. 
        '''

        super().__init__()

        if len(filters)     != len(qcircuits):   raise Exception(f"lenght of filters must be the same of lenght of qcircuits, found {len(filters)} and {len(qcircuits)}")
        if len(kernelsizes) != len(qcircuits):   raise Exception(f"lenght of kernelsizes must be the same of lenght of qcircuits, found {len(kernelsizes)} and {len(qcircuits)}")
        if len(strides)     != len(qcircuits):   raise Exception(f"lenght of strides must be the same of lenght of qcircuits, found {len(strides)} and {len(qcircuits)}")
        if len(filters)     != len(kernelsizes): raise Exception(f"lenght of filters must be the same of lenght of kernelsizes, found {len(filters)} and {len(kernelsizes)}")
        if len(filters)     != len(strides):     raise Exception(f"lenght of filters must be the same of lenght of strides, found {len(filters)} and {len(strides)}")
        if len(strides)     != len(kernelsizes): raise Exception(f"lenght of strides must be the same of lenght of kernelsizes, found {len(strides)} and {len(kernelsizes)}")
        if len(in_shape)    != 3:                raise Exception(f"length of in_shape must be equals to 3 (widht, height, channels, found {len(in_shape)}")
        
        self.encoder  = []
        self.decoder  = []
        self.sizes    = []
        
        w, h, c       = in_shape
        self.depth    = len(qcircuits)
        
        # Building the quanvolution encoder
        for i in range(self.depth):
            self.encoder.append(Quanvolution2D(qcircuit=qcircuits[i], filters=filters[i], kernelsize=kernelsizes[i], stride=strides[i]))
        
        filters.reverse()
        filters.append(filters[-1])
        kernelsizes.reverse()
        strides.reverse()

        # Building the classical decoder
        for i in range(self.depth):
            self.decoder.append(torch.nn.ConvTranspose2d(filters[i], filters[i+1], kernel_size=kernelsizes[i], stride=strides[i]))
        

    def encoder_f(self, x : torch.Tensor) -> torch.Tensor:
        '''
            Torch forward method for the encoder

            Parameters:  
            -----------
            - x : torch.Tensor  
                input for the torch model  

            Returns:  
            --------  
            - out : torch.Tensor  
                output from the torch model  
        '''
        
        for i in range(self.depth):
            x = torch.nn.functional.relu(self.encoder[i](x))
        
        out = x
        return out

    def decoder_f(self, x : torch.Tensor) -> torch.Tensor:
        '''
            Torch forward method for the decoder

            Parameters:  
            -----------
            - x : torch.Tensor  
                input for the torch model  

            Returns:  
            --------  
            - out : torch.Tensor  
                output from the torch model  
        '''

        for i in range(self.depth - 1):
            x = torch.nn.functional.relu(self.decoder[i](x))
        out = torch.nn.functional.sigmoid(self.decoder[-1](x))
        
        return out

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

        dec = self.encoder_f(x)
        out = self.decoder_f(dec)
        
        return out
        
class HybridAutoencoder(torch.nn.Module): 
    '''
        The encoder and the decoder are classical, the hidden space is processed by 
        a quantum circuit.
    '''

    def __init__(self):
        super().__init__()
        