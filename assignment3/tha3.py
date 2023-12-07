"""Introduction to Machine Learning THA3
"""

import numpy as np
import pandas as pd
from numpy import ndarray, float32

class Layer():
    "Class representing the weights and biases of a layer of the MLP using numpy"
    def __init__(self, shape:tuple[int]) -> None:
        """MLP Layer contructor
        
        Args:
            shape `<tuple[int]>`: shape of the weights/biases
        """

        self.weights:ndarray[float32] = np.zeros(shape=shape, dtype=float32)
        self.biases:ndarray[float32] = np.zeros(shape=(1, shape[0]), dtype=float32)

    def compute(self, x:ndarray[float32]) -> ndarray[float32]:
        """Calculating `wx + b`
        
        Args:
            x `<ndarray[float32]>`: input data
        """

        return np.add(np.dot(x, np.transpose(self.weights)), self.biases)

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
        self.h0:Layer = Layer((10,2))
        self.h1:Layer = Layer((10,10))
        self.o:Layer = Layer((2,10))

    def feed_forward(self, x:ndarray[float32]) -> float32:
        """One cycle of feed forward.
        
        Args:
            x `<ndarray[float32]>: One batch of training data.
            
        Returns:
            y_pred `<float32>`: Predicted y value."""
        # Layer 1
        net:ndarray[float32] = self.h0.compute(x)
        net:ndarray[float32] = self.h0.sigmoid(net)
        # Layer 2
        net:ndarray[float32] = self.h1.compute(net)
        net:ndarray[float32] = self.h1.sigmoid(net)
        # Output layer
        net:ndarray[float32] = self.o.compute(net)
        net:ndarray[float32] = self.o.sigmoid(net)

        return net[0]

    # TODO
    def back_propagation(self) -> None:
        """One cycle of back propagation.
        
        Args:
            e `<float32>`: error            
        """

df:pd.DataFrame = pd.read_excel("THA3train.xlsx")
mlp = MLP()
for ind in df.index:
    # input processing
    inp = ndarray(shape=(1,2))
    inp = np.array([df["X_0"][0],df["X_1"][0]])
    # feed forward
    y_pred = mlp.feed_forward(inp)
    # back propagation TODO
