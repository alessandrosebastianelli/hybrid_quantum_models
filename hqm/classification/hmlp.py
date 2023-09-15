import sys
sys.path += ['.', './layers/']

from hqm.layers.basiclayer import BasicLayer
import torch

class BasicHybridMLPClassifier(torch.nn.Module):
    '''
        This class implements a basic hybrid multilayer perceptron for classification purposes.
        BasicHybridMLPClassifier is composed of quantum layers stacked between two fully connected layers.
        The size of fully connected layers is set by means of in_dim and ou_dim.
    '''

    def __init__(self, qlayer : BasicLayer, in_dim : int, ou_dim : int) -> None:
        '''
            BasicHybridMLPClassifier constructor.  

            Parameters:  
            -----------  
            - qlayer : hqm.layers.basiclayers.BasicLayer  
                hqm quantum layer to be stacked between two fully connected layers  
            - in_dim : int  
                integer representing the input size for the first fully connected layer  
            - ou_dim : int   
                integer representing the output size of the hybrid model  
            
            Returns:  
            --------  
            Nothing, a BasicHybridMLPClassifier object will be created.    
        '''
        super().__init__()

        if in_dim < 1: raise Exception(f"The parameter in_dim must be greater than 1, found {in_dim}")
        if ou_dim < 1: raise Exception(f"The parameter ou_dim must be greater than 1, found {ou_dim}")

        n_qubits     = qlayer.n_qubits
        self.fc_1    = torch.nn.Linear(in_dim, n_qubits)
        self.qc_1    = qlayer.qlayer
        self.fc_2    = torch.nn.Linear(n_qubits, ou_dim)
        self.tanh    = torch.nn.Tanh()
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
        x = self.fc_1(x)
        x = self.tanh(x)
        x = self.qc_1(x)
        x = self.tanh(x)
        x = self.fc_2(x)
        out = self.softmax(x)
        return out
    
class MultiHybridMLPClassifier(torch.nn.Module):
    '''
        This class implements a hybrid multilayer perceptron with multiple quantum circuits for classification purposes.
        MultiHybridMLPClassifier is composed of several quantum layers stacked between two fully connected layers.
        The size of fully connected layers is set by means of in_dim and ou_dim.
    '''

    def __init__(self, qlayers : list, in_dim : int, ou_dim : int) -> None:
        '''
            MultiHybridMLPClassifier constructor.  

            Parameters:  
            -----------  
            - qlayers : list  
                list of hqm quantum layers to be stacked between two fully connected layers  
            - in_dim : int  
                integer representing the input size for the first fully connected layer  
            - ou_dim : int  
                integer representing the output size of the hybrid model  
            
            Returns:  
            --------  
            Nothing, a MultiHybridMLPClassifier object will be created.    
        '''

        super().__init__()

        if in_dim < 1: raise Exception(f"The parameter in_dim must be greater than 1, found {in_dim}")
        if ou_dim < 1: raise Exception(f"The parameter ou_dim must be greater than 1, found {ou_dim}")
        if len(qlayers) < 1: raise Exception(f"Size of qlayers must be greater than 1, found {len(qlayers)}")

        n_qubits_0   = qlayers[0].n_qubits
        n_qubits_1   = qlayers[-1].n_qubits
        self.fc_1    = torch.nn.Linear(in_dim, n_qubits_0)
        self.qcs     = [circ.qlayer for circ in qlayers]
        self.fc_2    = torch.nn.Linear(n_qubits_1, ou_dim)
        self.tanh    = torch.nn.Tanh()
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

        x = self.fc_1(x)
        x = self.tanh(x)
        for qc in self.qcs: 
            x = qc(x)
            x = self.tanh(x)
        x = self.fc_2(x)
        out = self.softmax(x)
        return out

class MultiHybridMultiMLPClassifier(torch.nn.Module):
    '''
        This class implements a hybrid multilayer perceptron with multiple quantum circuits for classification purposes.
        MultiHybridMultiMLPClassifier is composed of several quantum layers stacked between alternating fully connected layers.
        The size of fully connected layers is set by means of in_dim and ou_dim.
    '''

    def __init__(self, qlayers : list, in_dims : list, ou_dim : list) -> None:
        '''
            MultiHybridMultiMLPClassifier constructor.  

            Parameters:  
            -----------  
            - qlayers : list  
                list of hqm quantum layers to be stacked between two fully connected layers  
            - in_dims: list  
                list of integers representing the input size for the i-th fully connected layer (first value should correspond to size of input data)  
            - ou_dim : list  
                list of integers representing the output size for the i-th fully connected layer (last value should correspond to desired output size)  
            
            Returns:  
            --------  
            Nothing, a MultiHybridMultiMLPClassifier object will be created.    
        '''
        
        super().__init__()

        if len(in_dims) < 1: raise Exception(f"Size in_dims must be greater than 1, found {len(in_dims)}")
        if ou_dim < 1: raise Exception(f"The parameter ou_dim must be greater than 1, found {ou_dim}")
        if len(qlayers) < 1: raise Exception(f"Size of qlayerss must be greater than 1, found {len(qlayers)}")
        if len(qlayers) != len(in_dims): raise Exception(f"qlayers and in_dims must have the same lenght, found {len(qlayers)} and {len(in_dims)}")
        for i, dim in enumerate(in_dims): 
            if dim < 1: raise Exception(f"Element {i} of in_dims must be greater than 1, found {dim}")
            else: pass
        
        self.fcs     = [torch.nn.Linear(dim, circ.n_qubits) for (dim, circ) in zip(in_dims, qlayers)]
        self.fco     = torch.nn.Linear(qlayers[-1].n_qubits, ou_dim)
        self.qcs     = [circ.qlayer for circ in qlayers]
        self.tanh    = torch.nn.Tanh()
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
            - x : torch.Tensor  
                output from the torch model  
        '''
        
        for fc, qc in zip(self.fcs, self.qcs):
            x = fc(x)
            x = self.tanh(x)
            x = qc(x)
            x = self.tanh(x)
        
        x   = self.fco(x) 
        out = self.softmax(x)
        return out