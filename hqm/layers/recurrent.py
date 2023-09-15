import torch
import sys

sys.path += ['.', './utils/', '/circuits/']

from hqm.circuits.circuit import QuantumCircuit
from hqm.utils.aiinterface import AIInterface


class QGRU(torch.nn.Module):
    '''
        Quantum Gradient Recurrent Unit layer.

        Currently supports only Torch.

        Reference
        ---------
        A. Ceschini, A. Rosato and M. Panella, "Hybrid Quantum-Classical Recurrent  
        Neural Networks for Time Series Prediction," 2022 International Joint Conference  
        on Neural Networks (IJCNN), Padua, Italy, 2022, pp. 1-8,   
        doi: 10.1109/IJCNN55064.2022.9892441.
    '''

    def __init__(self, qcircuits : list[QuantumCircuit], inputsize : int, hiddensize : int, aiframework : str = 'torch') -> None:
        '''
        QGRU constructor.  

        Parameters:  
        -----------   
        - qcircuits : list of QuantumCircuit  
            list containing three quantum circuits in this exact order: 1) Quantum Layer Reset, 2) Quantum Layer Update, 3) Quantum Layer Output
        - inputsize : int  
            integer representing the number of variable (channels) in the input date
        - hiddensize : int  
            integer size representing the recurrent filters
        - aiframework : str  
            string representing the AI framework in use, can be 'torch' or 'keras'. This will create  
            a compatible trainable layer for the framework.

        Returns:    
        --------     
        Nothing, a QGRU object will be created.  
        '''
        
        super().__init__()

        if aiframework not in ['torch', 'keras']: raise Exception(f"Quanvolution2D curently supports only 'torch' as framework, found {aiframework}")
        if inputsize       < 1:                   raise Exception(f"inputsize must be greater than 1, found {inputsize}")
        if hiddensize      < 1:                   raise Exception(f"hiddensize must be greater than 1, found {hiddensize}")
        if len(qcircuits) != 3:                   raise Exception(f"qcircuits must contain 3 elements, one for reset gate, one for update gate and one for output gate, found {len(qcircuits)}")

        if (qcircuits[0].n_qubits != qcircuits[1].n_qubits) or (qcircuits[0].n_qubits != qcircuits[2].n_qubits) or (qcircuits[1].n_qubits != qcircuits[2].n_qubits) or (qcircuits[1].n_qubits != qcircuits[2].n_qubits):
            raise Exception(f"n_qubits must be the same for each circuit in qcircuits, found {qcircuits[0].n_qubits}, {qcircuits[1].n_qubits} and {qcircuits[2].n_qubits}")

        self.aiframework    = aiframework
        self.n_qubits       = qcircuits[0].n_qubits
        self.hiddensize     = hiddensize

        
        self.clayer_in      = torch.nn.Linear(inputsize+hiddensize, self.n_qubits)
        self.clayer_out     = torch.nn.Linear(self.n_qubits, hiddensize)

        self.qlayer_reset   = AIInterface.network_layer(
                                circuit      = qcircuits[0].circuit, 
                                weight_shape = qcircuits[0].weight_shape, 
                                n_qubits     = self.n_qubits, 
                                aiframework  = self.aiframework
                            )

        self.qlayer_update  = AIInterface.network_layer(
                                circuit      = qcircuits[1].circuit, 
                                weight_shape = qcircuits[1].weight_shape, 
                                n_qubits     = self.n_qubits, 
                                aiframework  = self.aiframework
                            )
        
        self.qlayer_output  = AIInterface.network_layer(
                                circuit      = qcircuits[2].circuit, 
                                weight_shape = qcircuits[2].weight_shape, 
                                n_qubits     = self.n_qubits, 
                                aiframework  = self.aiframework
                            )
        
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Torch forward function for QGRU layer

        Parameters:
        -----------
        - x : torch.Tensor
            input image or tensor
        
        Returns:
        --------
        - out : torch.Tensor
            qgru input
        '''

        if len(x.shape) != 3: raise Exception(f"x must be a tensor of 3 elements (batch, sequencelenght, featuressize), found {len(x.shape)}")

        batch_size, seq_length, featuressize = x.size()            
        hidden_seq = []
        h_t = torch.zeros(batch_size, self.hiddensize)
        
        for t in range(seq_length):
             # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]

            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)

            # match qubit dimension
            y_t = self.clayer_in(v_t)

            r_t = torch.sigmoid(self.clayer_out(self.qlayer_reset(y_t)))  # forget block
            z_t = torch.sigmoid(self.clayer_out(self.qlayer_update(y_t)))  # update block

            # Concatenate input and hidden state
            v2_t = torch.cat(((r_t * h_t), x_t), dim=1)

            # match qubit dimension
            y2_t = self.clayer_in(v2_t)
            
            h_tilde_t = torch.tanh(self.clayer_out(self.qlayer_output(y2_t)))

            h_t = ((1-z_t) * h_tilde_t) + (z_t * h_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        out = hidden_seq
        return out