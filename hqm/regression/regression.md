# Regression

## hmlp - hybrid multi-layer perceptron

This package contains several implementations of hybrid multi-layer perceptron.

### BasicHybridMLPRegressor
This class implements a basic hybrid multilayer perceptron for regression purposes. BasicHybridMLPRegressor is composed of quantum layers stacked between two fully connected layers. The size of fully connected layers is set by means of in_dim and ou_dim.

<img src="../regression/BasicHybridMLPRegressor.png" width="700" height="350" />
    
### MultiHybridMLPRegressor
This class implements a hybrid multilayer perceptron with multiple quantum circuits for regression purposes. MultiHybridMLPRegressor is composed of several quantum layers stacked between two fully connected layers. The size of fully connected layers is set by means of in_dim and ou_dim.

<img src="../regression/MultiHybridMLPRegressor.png" width="700" height="350" />
    

### MultiHybridMultiMLPRegressor
This class implements a hybrid multilayer perceptron with multiple quantum circuits for regression purposes. MultiHybridMultiMLPRegressor is composed of several quantum layers stacked between alternating fully connected layers. The size of fully connected layers is set by means of in_dim and ou_dim.

<img src="../regression/MultiHybridMultiMLPRegressor.png" width="900" height="350" />
    