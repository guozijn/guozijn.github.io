---
title: Understanding Single-Layer Neural Networks
tags:
  - machine learning
  - neural network
---
## Key Concepts
1. Neuron (Perceptron)  
    A neutron is the fundamental unit of the network. Each neuron computes a weighted sum of inputs plus a bias, then applies an activation function (e.g. step, sigmoid)
2. Input Layer  
    The input layer accepts the raw data features. Each input is multiplied by an associated weight and passed to the neuron.
3. Weights and Bias  
    - Weights represent the importance of each input feature.
    - Bias allows shifting the decision boundary away from the origin.
4. Linear Combination  
    The neuron computes:

    $$
    z = \sum_{i=1}^{n} w_i x_i + b
    $$

    Where $w_i$ = weight, $x_i$ = input, $b$ = bias
5. Activation Function  
    The activation function introduces non-linearity (in classic perceptron often a step function). It determines the output class or value. Without it, the network is just a linear model.  
6. Output Layer. 
    It provides the final prediction. In a single-layer network, there is only one set of weights between the input and output (no hidden layer).
7. Decision Boundary.  
    The hyperplane separating classes in the input space. In a single-layer network, this boundary is always linear.

## Architecture of a Single-Layer Neural Network
![single_layer_nn_binary.png](https://images.zijianguo.com/single_layer_nn_binary.png)
   
   
  
