# SARSA( &lambda; ) with a Neural Network

### Description
- SARSA(&lambda;) with a neural network with one hidden layer. The input of the neural network is the one-hot encoding of the state number. We use the one-hot encoding of the state number instead of the state number itself because we do not want to build the prior knowledge that integer number inputs close to each other have similar values. 

- The hidden layer contains 100 rectifier linear units (ReLUs) which pass their input if it is bigger than 1 else return 0 otherwise. 

- ReLU gates are commonly used in neural networks due to their nice properties such as the sparsity of the activation and having non-vanishing gradients. The output of the neural network is the estimated state value. It is a linear function of the hidden units as is commonly the case when estimating the value of a continuous target using neural networks.
