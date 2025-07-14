import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.autolayout'] = True

# Neural Network Class
class AI:

    # Initialize the number of inputs, outputs, and epochs
    def __init__(self, inputs, outputs, epochs=50):
        self.M = inputs
        self.N = outputs
        self.epochs = range(epochs)

    # Trains the neural network and plots the weights of the neural network
    def __call__(self, ax, inx, outy):

        # Convert lists to numpy arrays
        x, y = np.array(inx), np.array(outy)
        bias = -1

        # Training epochs
        for epoch in self.epochs:

            # Forward Propigation using the Sigmoid Function and bias
            for i in self.axis:
                if i == self.axis[0]:
                    self.layers[i] = x @ self.weights[i] + bias
                    self.slayers[i] = self.sigmoid(self.layers[i])
                else:
                    self.layers[i] = self.slayers[i+1] @ self.weights[i] + bias
                    self.slayers[i] = self.sigmoid(self.layers[i])

            # Backpropigation where deltas adjusting neural network weights are stored in dx
            dx = {}
            for i in self.raxis:
                # Computing the error and delta for each layer
                if i == self.raxis[0]:
                    # First error
                    error = (y - self.slayers[i])**2
                    delta = 2.0*(y - self.slayers[i])*self.sigmoid(self.layers[i], derv=True)
                    dx[i] = delta
                else:
                    # Layers are passed into the sigmoid derivative
                    error = self.weights[i-1] @ delta
                    delta = error * self.sigmoid(self.layers[i], derv=True)
                    dx[i] = delta

            # Updates the neural network weights with their respective layer delta
            for i in self.axis:
                self.weights[i] -= dx[i]

        # Animated plot of all layers weights during training
        for p, i in enumerate(self.axis):
            z = self.weights[i]
            mm, nn = z.shape
            xx, yy = np.meshgrid(range(nn), range(mm))
            ax[p].cla()
            ax[p].set_title(f'Weights: {p+1}')
            ax[p].contourf(xx, yy, z, cmap='viridis')

        plt.pause(0.001)

    # Sigmoid activation function
    def sigmoid(self, x, derv=False):
        f = 1.0 / (1.0 + np.exp(-x))
        if derv:
            return f*(1 - f)
        return f

    # Build the neural network in a linear formation and initalize weights and layers
    def build(self):
        M, N = self.M, self.N 
        self.axis = list(range(M, N, -1))
        self.raxis = self.axis[::-1]

        self.weights = {}
        self.layers = {}
        self.slayers = {}

        for i in self.axis:
            self.weights[i] = 0.05*np.random.rand(i, i-1)
            self.layers[i] = np.zeros(i-1)
            self.slayers[i] = np.zeros(i-1)

# Generate Matplotlib figure
fig = plt.figure(figsize=(10, 7))
plots = [fig.add_subplot(2, 3, i) for i in range(1, 7)]

# Declare neural network parameters
ai = AI(9, 3, epochs=100)
ai.build()

# Build a random dataset input and output list
rows = 200
dataset = np.random.rand(rows, 9)
output = np.random.rand(rows, 3)

# Pass the data to train the model and visualize weights
for index, (xx, yy) in enumerate(zip(dataset, output)):
    print("Number of rows left: ", rows - index)
    ai(plots, xx, yy)


plt.show()
