
import torch


class QuanvolutionAutoencoder(torch.nn.Module): 
    '''
        The encoder is composed of quanvolution layer and the encoder is classical
    '''
    pass

class HybridAutoencoder(torch.nn.Module): 
    '''
        The encoder and the decoder are classical, the hidden space is processed by 
        a quantum circuit.
    '''
    pass