# Quantum Circuits
This package contains the implementations for basic, advanced and custom quantum circuits based on PennyLane. The implementation consists of wrappers that encapsulate PennyLane circuits in order to make them compatible with Keras or Pytorch. 

## Angle Encoding

This section relates to circuits that use angle embedding as a feature embedding strategy. This is done by means of the PennyLane operator *AngleEmbedding*.

This operator encodes $N$ features into the rotation angles of $n$ qubits, where $N\leq n$. The length of features has to be smaller or equal to the number of qubits. If there are fewer entries in features than rotations, the circuit does not apply the remaining rotation gates.

The following circuits have this operator as the first layer.

[PennyLane Documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.AngleEmbedding.html)


### Basic Entangler Circuit

Layers consisting of one-parameter single-qubit rotations on each qubit, followed by a closed chain or ring of CNOT gates.

The ring of CNOT gates connects every qubit with its neighbor, with the last qubit being considered as a neighbor to the first qubit.


<img src="../circuits/basic_entangler_circuit.png" width="80%"/>

When using a single wire, the template only applies the single qubit gates in each layer.

[PennyLane Documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.BasicEntanglerLayers.html)

### Strongly Entangling Circuit

Layers consisting of single qubit rotations and entanglers, inspired by the circuit-centric classifier design arXiv:1804.00633.

<img src="../circuits/strongly_entangler_circuit.png" width="80%"/>

[PennyLane Documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.StronglyEntanglingLayers.html)

### Random Circuit

Layers of randomly chosen single qubit rotations and 2-qubit entangling gates, acting on randomly chosen qubits.

<img src="../circuits/random_circuit.png" width="80%"/>

[PennyLane Documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.RandomLayers.html)


## Data Reuploading
