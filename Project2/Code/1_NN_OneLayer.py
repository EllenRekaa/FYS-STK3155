# to categorical turns our integer vector into a onehot representation
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def feed_forward(X):
        
    # weighted sum of inputs to the hidden layer
    z_h = np.matmul(X, hidden_weights) + hidden_bias
    # activation in the hidden layer
    a_h = sigmoid(z_h)

    # weighted sum of inputs to the output layer
    z_o = np.matmul(a_h, output_weights) + output_bias
    # softmax output
    # axis 0 holds each input and axis 1 the probabilities of each category

    exp_term = np.exp(z_o)
    #print(np.shape(exp_term))
    probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
    print(np.shape(probabilities))    
    # for backpropagation need activations in hidden and output layers
    return a_h, probabilities

def backpropagation(X, Y):
    a_h, probabilities = feed_forward(X)  
    p = np.argmax(probabilities, axis=1)
    # error in the output layer
    #error_output = p - np.reshape(Y,(len(Y),1))
    error_output = pred - Y
    
    # error in the hidden layer
    error_hidden = np.matmul(error_output, output_weights.T) * a_h * (1 - a_h)
    
    # gradients for the output layer
    output_weights_gradient = np.matmul(a_h.T, error_output)
    output_bias_gradient = np.sum(error_output, axis=0)
    
    # gradient for the hidden layer
    hidden_weights_gradient = np.matmul(X.T, error_hidden)
    hidden_bias_gradient = np.sum(error_hidden, axis=0)

    return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient

# we obtain a prediction by taking the class with the highest likelihood
def predict(X):
    probabilities = feed_forward(X)
    return np.argmax(probabilities, axis=1)


# ensure the same random numbers appear every time
np.random.seed(0)

# data:
# Design matrix
X = np.array([ [0, 0], [0, 1], [1, 0],[1, 1]],dtype=np.float64)

# The XOR gate
yXOR = np.array( [ 0, 1 ,1, 0])
# The OR gate
yOR = np.array( [ 0, 1 ,1, 1])
# The AND gate
yAND = np.array( [ 0, 0 ,0, 1])
    
  
# Split data
#X_train, X_test, Y_train, Y_test = train_test_split(X,yOR, train_size=0.8)
X_train = X
Y_train = yOR

# Defining the neural network
n_inputs, n_features = X_train.shape # (4,2) 4 sett med (x1,x2)
n_hidden_neurons = 2 # 
n_categories = 2 # 2 output (0,1)
n_features = 2 # x1 og x2


# we make the weights normally distributed using numpy.random.randn
# weights and bias in the hidden layer
hidden_weights = np.random.randn(n_features, n_hidden_neurons)
hidden_bias = np.zeros(n_hidden_neurons) + 0.01

# weights and bias in the output layer
output_weights = np.random.randn(n_hidden_neurons, n_categories)
output_bias = np.zeros(n_categories) + 0.01

#print("New accuracy on training data: " + str(accuracy_score(predict(X_train), Y_train)))

eta = 0.01
lmbd = 0.01
#loop over etas and lambdas

# loop over epochs
for i in range(1000):
    # calculate gradients
    dWo, dBo, dWh, dBh = backpropagation(X_train, Y_train)
    
    # regularization term gradients
    dWo += lmbd * output_weights
    dWh += lmbd * hidden_weights
    
    # update weights and biases
    output_weights -= eta * dWo
    output_bias -= eta * dBo
    hidden_weights -= eta * dWh
    hidden_bias -= eta * dBh

"""    
    loss = -1 *np.sum( y_pred *log(propability +(1-y)) * log(1-a))

print("New accuracy on training data: " + str(accuracy_score(predict(X_train), Y_train)))
"""