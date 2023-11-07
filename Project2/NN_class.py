import numpy as np
import matplotlib.pyplot as plt

def sigmoid(input):

    term = 1.0 / (1 - np.exp(-input))
    return term

def sigmoid_prime(input):

    return sigmoid(input) * (1 - sigmoid(input))

def tanh(input):

    return np.tanh(input)

def tanh_prime(input):

    return 1 - np.tanh(input) ** 2

def relu(input):
    zero = np.zeros(input.shape)
    return np.max([input,zero], axis=0)
def relu_prime(input):
    return 1.0

def identity(input):
    return input

def identity_prime(input):
    return 1.0

def MSE(y_in, y_pred):
    return np.mean(np.power(y_in - y_pred, 2))

def MSE_prime(y_in, y_pred):
    return 2 * (y_pred - y_in) / y_in.size




class SingleLayer:

    def __init__(self):

        self.input = None
        self.output = None


    def forwardpass(self, input):

        raise NotImplementedError

    def backpropagation(self, output_error, learning_schedule, lmd):

        raise NotImplementedError


class ConnectLayers(SingleLayer):

    def __init__(self, input_dim, output_dim):

        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.random.randn(1, output_dim)

    def forwardpass(self, data):

        self.input = data
        self.output = np.dot(self.input, self.weights) + self.bias

        return self.output

    def backpropagation(self, output_error, learning_schedule,lmd):

        input_error = np.dot(output_error, self.weights.T)

        print(self.input.T)
        print(output_error)
        weights_error = self.input.T @ output_error
        bias_error = np.sum(output_error, axis=0)

        #self.weights += lmd * self.weights

        self.weights -= learning_schedule * weights_error
        self.bias -= learning_schedule * bias_error

        return input_error

class ActivationLayer(SingleLayer):
    def __init__(self, activation, activation_prime):

        self.activation = activation
        self.activation_prime = activation_prime

    def forwardpass(self, data):

        self.input = data
        self.output = self.activation(self.input)

        return self.output

    def backpropagation(self, output_error, learning_schedule,lmd):
        return self.activation_prime(self.input) * output_error


class NeuralNetwork:

    def __init__(self):

        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):

        self.layers.append(layer)

    def set_cost_function(self, loss, loss_prime):

        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, data):


        samples = len(data)
        result = []

        for i in range(samples):
            output = data[i]
            for layer in self.layers:
                output = layer.forwardpass(output)

            result.append(output)

        return result

    def train(self, X, Y, epochs, learning_schedule, lmd):

        samples = len(X)

        for i in range(epochs):
            err = 0.0
            for j in range(samples):
                output = X[j]

                for layer in self.layers:
                    output = layer.forwardpass(output)


                err += self.loss(Y[j], output)


                error = self.loss_prime(Y[j], output)

                for layer in reversed(self.layers):
                    error = layer.backpropagation(error, learning_schedule, lmd)

            err /= samples

            #print('epoch %d/%d  error = %f' % (i+1, epochs, err))

np.random.seed(0)
n = 5
maxd = 3
x = np.linspace(-1,1,n)

Y = 3 + x ** 2 + np.random.normal(0,1,x.shape)


X = np.array((n,1))



"""
X = np.array([ [0, 0], [0, 1], [1, 0],[1, 1]],dtype=np.float64)
Y = np.array([0,1,1,0])
"""
"""
X = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
Y= np.array([[[0]], [[1]], [[1]], [[0]]])
"""



net = NeuralNetwork()

net.add(ConnectLayers(n,5))
net.add(ActivationLayer(identity, identity_prime))
net.add(ConnectLayers(n,1))
net.add(ActivationLayer(identity, identity_prime))

net.set_cost_function(MSE, MSE_prime)

lrs = np.logspace(-6,1,10)
lmds = np.logspace(-10,1,10)
accuracy = np.zeros((len(lrs), len(lmds)))
for i in range(len(lrs)):
    for j in range(len(lmds)):
        net.train(X, Y, epochs = 100, learning_schedule = lrs[i], lmd = lmds[j])
        accuracy[i][j] = MSE(Y,net.predict(X))

print(accuracy)


































"""
import numpy as np

class NeuralNetwork:
    def __init__(
            self,
            X_dat,
            Y_dat,
            n_neurons = 2,
            n_cat = 1,
            epochs = 1000,
            batch_size = 100,
            eta = 0.01,
            lmd = 0.0):

        self.n_inputs = X_dat.shape[0]
        self.n_features = X_dat.shape[1]
        self.n_neurons = n_neurons
        self.n_cat = n_cat

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmd = lmd

        self.create_network()

    def create_network(self):

        self.hidden_weights = np.random.randn(self.n_features, self.n_neurons)
        self.hidden_bias = np.random.randn(1,self.n_neurons)

        self.output_weights = np.random.randn(self.n_neurons, self.n_cat)
        self.output_bias = np.random.randn(1, self.n_cat)

    def forwardpass(self):

        self.IN_TO_HIDDEN = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.HIDDEN_TRANSFER = self.activation_function(self.IN_TO_HIDDEN)

        self.HIDDEN_TO_OUT = np.matmul(self.HIDDEN_TRANSFER, self.output_weights) + self.output_bias

        exp_term = np.exp(self.HIDDEN_TO_OUT)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims = True)

    def forwardpass_to_out(self, X):

        IN_TO_HIDDEN = np.matmul(X, self.hidden_weights) + self.hidden_bias
        HIDDEN_TRANSFER = self.activation_function(IN_TO_HIDDEN)
        HIDDEN_TO_OUT = np.matmul(HIDDEN_TRANSFER, self.output_weights) + self.output_bias

        exp_term = np.exp(HIDDEN_TO_OUT)
        prob = exp_term/np.sum(exp_term, axis=1, keepdims=True)

        return prob

    def backpropagation(self):

        error_output = self.probabilities - self.Y_dat
        term = self.HIDDEN_TRANSFER * (1 - self.HIDDEN_TRANSFER)
        error_hidden = np.matmul(error_output, self.output_weights) * term

        self.output_dW = np.matmul(self.HIDDEN_TRANSFER.T, error_output)
        self.output_dB = np.sum(error_hidden, axis=0)

        self.hidden_dW = np.matmul(self.X_dat.T, error_hidden)
        self.hidden_dB = np.sum(error_hidden, axis=0)
        #if regularization
        if self.lmd > 0.0:
            self.output_dW += self.lmd * self.output_weights
            self.hidden_dW += self.lmd * self.hidden_weights

        self.output_weights -= self.eta * self.output_dW
        self.output_bias -= self.eta * self.output_dB
        self.hidden_weights -= self.eta * self.hidden_dW
        self.hidden_bias -= self.eta * self.hidden_dB

    def activation_function(self, input):

        term = 1.0 /(1.0 + np.exp(-input))

        return term

    def predict(self, X):

        prob = self.forwardpass_to_out(X)
        return np.argmax(prob, axis=1)

    def train(self):
        idx = np.arange(self.n_inputs)

        for _ in range(self.epochs):
            for _ in range(self.iterations):
                rnd_idx = np.random.choice(idx, size=self.batch_size,replace=False)

                self.X_dat = self.X_dat[rnd_idx]
                self.Y_dat = self.Y_dat[rnd_idx]

                self.forwardpass()
                self.backpropagation()


epochs = 1000
batch_size = 5
lmd = 0.01
eta = 0.001
np.random.seed(0)
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector

n = 10
x = np.linspace(-1,1,n)
y = x*2
maxd = 3
#X = np.column_stack([x**i for i in range(maxd)])


X = np.array([ [0, 0], [0, 1], [1, 0],[1, 1]],dtype=np.float64)

yOR = np.array([0,1,1,1])
yAND = np.array([0,0,0,1])
yORHOT = to_categorical_numpy(yAND)

dnn = NeuralNetwork(X, yORHOT, eta=eta, lmd = lmd, epochs = epochs,
                    batch_size=batch_size, n_neurons = 2, n_cat = 1)
dnn.train()

print(dnn.predict(X))

"""
