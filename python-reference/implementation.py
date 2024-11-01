import numpy as np
import pandas as pd

class NeuralNet:
    def __init__(self, layers):
        """
        Parameters
        -----–----
        layers : list of integers
            The number of nodes in each layer
        """
        self.num_layers = len(layers)
        self.layers = layers
        self.biases = [np.random.randn(n) for n in layers[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(layers[1:], layers[:-1])]
    
    @staticmethod
    def σ(x):
        return 1/(1+np.exp(-x))
    @staticmethod
    def dσ(x):
        return np.exp(-x)/(1+np.exp(-x))**2

#     def cost_derivative(self, ans, activation):
#         return 2*(activation-(np.eye(self.layers[-1])[ans]))
    def cost_derivative(self, ans, activation):
        y = np.eye(self.layers[-1])[ans]
        return (activation-y)/(activation + 1e-15)/(1-activation+1e-15)
    
    def propagate_forward(self, vector):
        """Return the output of the neutral net of vector is the input"""
        for weights, biases in zip(self.weights, self.biases):
            vector = self.σ(weights @ vector + biases)
        return vector
    
    def train(self, training_data, iterations, batch_size, learning_rate):
        """
        Train the neural network using stochastic gradient descent
        
        Parameters
        ----------
        training_data : tuple(1D numpy array of answers, training inputs)
            The data used to train the network.
        iterations : int
            The number of times to loop over the entire data set for training.
        batch_size : int
            The number of inputs to average for each gradient step.
        learning_rate : float
            The size of the step to take.
        """
        answers, data = training_data
        n = len(answers)
        for i in range(iterations):
            shuffled_data = np.arange(n).reshape(n//batch_size, batch_size)
            np.random.shuffle(shuffled_data)
            for choice in shuffled_data:
                self.update_from_batch((answers[choice], data[choice]), learning_rate)

    def update_from_batch(self, batch, learning_rate):
        """Process a batch of training data"""
        d_weights = [np.zeros_like(w) for w in self.weights]
        d_biases = [np.zeros_like(b) for b in self.biases]
        
        answers, data = batch
        for ans, datum in zip(answers, data):
            dd_weights, dd_biases = self.propagate_backwards(ans, datum)
            d_weights = [dw+ddw for dw, ddw in zip(d_weights, dd_weights)]
            d_biases = [db+ddb for db, ddb in zip(d_biases, dd_biases)]
        self.weights = [w-learning_rate/len(answers)*dw 
                        for w, dw in zip(self.weights, d_weights)]
        self.biases = [b-learning_rate/len(answers)*db for b, db in zip(self.biases, d_biases)]
    
    def propagate_backwards(self, answer, data):
        """Using one piece of labelled data calulcate the gradient"""
        d_weights = [np.zeros_like(w) for w in self.weights]
        d_biases = [np.zeros_like(b) for b in self.biases]

        # Propagate forward
        activation = data
        activations = [data]
        zs = []
        for weights, biases in zip(self.weights, self.biases):
            z = weights @ activation + biases
            zs.append(z)
            activation = self.σ(z)
            activations.append(activation)
        
        # Propagate backwards
        for l in range(1, self.num_layers):
            if l == 1:
                delta = self.cost_derivative(answer, activations[-1])
            else:
                delta = self.weights[-l+1].T @ ((self.dσ(zs[-l+1])*delta))
            d_biases[-l] = self.dσ(zs[-l])*delta
            d_weights[-l] = (activations[-l-1][:,np.newaxis]*d_biases[-l]).T

        return d_weights, d_biases
    
    def evaluate(self, test_data):
        answers, data = test_data

        count = 0
        n = len(answers)
        for ans, datum in zip(answers, data):
            if np.argmax(self.propagate_forward(datum)) == ans:
                count += 1
        return count / n