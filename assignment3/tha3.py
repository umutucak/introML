"""Introduction to Machine Learning THA3
"""

import numpy as np
from numpy import ndarray, float32

class Layer():
    "Class representing the weights and biases of a layer of the MLP using numpy"
    def __init__(self, shape:tuple[int]) -> None:
        """MLP Layer contructor
        
        Args:
            shape `<tuple[int]>`: shape of the weights/biases
        """

        self.weights = ndarray(shape=shape, dtype=float32)
        self.biases = ndarray(shape=shape, dtype=float32)

    def compute(self, x:ndarray[float32]) -> ndarray[float32]:
        """Calculating `wx + b`
        
        Args:
            x `<ndarray[float32]>`: input data
        """

        # TODO we need to transpose here. how to know to transpose weights or x?
        return np.add(np.multiply(self.weights, x), self.biases)

    def sigmoid(self, x:ndarray[float32]) -> ndarray[float32]:
        """Putting the output of `Layer.compute()` into the Sigmoid activation function.
        
        Args:
            x `<ndarray[float32]`: `wx + b` vectorized
        
        Returns:
            `sig(wx+b)`
        """

        return np.divide(1, np.add(1, np.exp(-x)))

class MLP():
    """Class representing an MLP with the preconfiguration from the assignment."""
    def __init__(self) -> None:
        self.h0 = Layer((10,2))
        self.h1 = Layer((10,10))
        self.o = Layer((2,10))

    def feed_forward(self, x:ndarray[float32]) -> float32:
        """One cycle of feed forward.
        
        Args:
            x `<ndarray[float32]>: One batch of training data.
            
        Returns:
            y_pred `<float32>`: Predicted y value."""
        # Layer 1
        net = self.h0.compute(x)
        net = self.h0.sigmoid(net)
        # Layer 2
        net = self.h1.compute(net)
        net = self.h1.sigmoid(net)
        # Output layer
        net = self.o.compute(net)
        net = self.o.sigmoid(net)

        return net

    # TODO
    def back_propagation(self) -> None:
        """One cycle of back propagation.
        
        Args:
            e `<float32>`: error            
        """
