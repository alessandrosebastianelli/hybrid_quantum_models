# Classification

## hmlp - hybrid multi-layer perceptron

This package contains several implementations of hybrid multi-layer perceptron.

### BasicHybridMLPClassifier
This class implements a basic hybrid multilayer perceptron for classification purposes. BasicHybridMLPClassifier is composed of quantum layers stacked between two fully connected layers. The size of fully connected layers is set by means of in_dim and ou_dim.

<img src="../classification/BasicHybridMLPClassifier.png" width="700" height="350" />
    
### MultiHybridMLPClassifier
This class implements a hybrid multilayer perceptron with multiple quantum circuits for classification purposes. MultiHybridMLPClassifier is composed of several quantum layers stacked between two fully connected layers. The size of fully connected layers is set by means of in_dim and ou_dim.

<img src="../classification/MultiHybridMLPClassifier.png" width="700" height="350" />
    

### MultiHybridMultiMLPClassifier
This class implements a hybrid multilayer perceptron with multiple quantum circuits for classification purposes. MultiHybridMultiMLPClassifier is composed of several quantum layers stacked between alternating fully connected layers. The size of fully connected layers is set by means of in_dim and ou_dim.

<img src="../classification/MultiHybridMultiMLPClassifier.png" width="900" height="350" />

## hcnn - hybrid convolutional neural network

### HybridLeNet-5
