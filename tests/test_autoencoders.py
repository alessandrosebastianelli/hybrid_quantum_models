import sys
sys.path += ['.', './hqm/']

from hqm.encoding.autoencoders import QuanvolutionAutoencoder, HybridAutoencoder
from hqm.circuits.angleencoding import BasicEntangledCircuit
from hqm.layers.basiclayer import BasicLayer

import torch

def test_quanvolution_autoencoder_output():

    qcircuits = []
    qcircuits += [BasicEntangledCircuit(n_qubits=9, n_layers=1)]
    qcircuits += [BasicEntangledCircuit(n_qubits=9, n_layers=1)]
    qcircuits += [BasicEntangledCircuit(n_qubits=9, n_layers=1)]

    autoencoder = QuanvolutionAutoencoder(qcircuits=qcircuits, 
                                            in_shape=(16,16,3), 
                                            filters=[3,6,9], 
                                            kernelsizes=[3,3,3], 
                                            strides=[1,1,1])
    
    x = torch.rand(size=(1,3,16,16))

    o = autoencoder(x)

    assert x.shape == o.shape

def test_hybrid_autoencoder_output():

    qcircuit  = BasicEntangledCircuit(n_qubits=9, n_layers=1)
    qlayer    = BasicLayer(qcircuit=qcircuit, aiframework='torch')

    autoencoder = HybridAutoencoder(qlayer=qlayer, 
                                    in_shape=(16,16,3), 
                                    filters=[3,6,9], 
                                    kernelsizes=[3,3,3], 
                                    strides=[1,1,1])
    
    x = torch.rand(size=(1,3,16,16))
    o = autoencoder(x)

    assert x.shape == o.shape
  




