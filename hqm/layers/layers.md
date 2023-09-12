# Quantum Layers
This package contains the implementations for quantum layers, in other words, deep learning layers that contain user-defined quantum circuits.

## Basic Layer
This class serves as a foundational layer without any specific functionalities, primarily designed to facilitate the integration of quantum circuits with Torch or Keras layers. Its main purpose is to establish a bridge between the quantum computing framework and deep learning libraries such as Torch or Keras.


## Quanvolution Layers
Derived from the basic layer, this specific layer is designed to perform quantum convolutions, often referred to as "quanvolutions." It's important to note that, as of now, this layer is exclusively accessible and compatible with Torch-based models.

<img src="../layers/quanvolution.png" width="80%"/>

**Reference**  

Henderson, M., Shakya, S., Pradhan, S., & Cook, T. (2020). Quanvolutional neural networks: powering image recognition with quantum circuits. Quantum Machine Intelligence, 2(1), 2.


## Recurrent Layer
Derived from the basic layer, this particular layer serves as an implementation of a recurrent neural layer. It's essential to note that, at the moment, this layer is exclusively accessible and compatible with Torch-based models.

<img src="../layers/qgru.png" width="80%"/>


**Reference**  
  
A. Ceschini, A. Rosato and M. Panella, "Hybrid Quantum-Classical Recurrent Neural Networks for Time Series Prediction," 2022 International Joint Conference on Neural Networks (IJCNN), Padua, Italy, 2022, pp. 1-8, doi: 10.1109/IJCNN55064.2022.9892441.