# to categorical turns our integer vector into a onehot representation
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

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
    probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    
    # for backpropagation need activations in hidden and output layers
    return a_h, probabilities

def backpropagation(X, Y):
    a_h, probabilities = feed_forward(X)  
    #p = np.argmax(probabilities, axis=1)
    # error in the output layer
    error_output = probabilities - Y
    #error_output = pred - Y
    
    # error in the hidden layer
    error_hidden = np.matmul(error_output, output_weights.T) * a_h * (1 - a_h)
    
    # gradients for the output layer
    output_weights_gradient = np.matmul(a_h.T, error_output)
    output_bias_gradient = np.sum(error_output, axis=0)
    
    # gradient for the hidden layer
    hidden_weights_gradient = np.matmul(X.T, error_hidden)
    hidden_bias_gradient = np.sum(error_hidden, axis=0)

    return a_h, probabilities, output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient

# we obtain a prediction by taking the class with the highest likelihood
def predict(X):
    a, probabilities = feed_forward(X)
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
X_train, X_test, Y_train, Y_test = train_test_split(X,yOR, train_size=0.8)
Y_train = np.reshape(Y_train,(len(Y_train),1))
#X_train = X
#Y_train = np.reshape(yOR,(len(yOR),1))

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

print("New accuracy on training data: " + str(accuracy_score(predict(X_train), Y_train)))

eta = 0.01
lmbd = 0.01
eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)
end= 100
lim = 0.01
Loss = np.zeros(end)
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

#loop over etas and lambdas
for i, eta in enumerate(eta_vals):
   for j, lmbd in enumerate(lmbd_vals):
        # loop over epochs
        for k in range(end):
            # calculate gradients
            a_h,P, dWo, dBo, dWh, dBh = backpropagation(X_train, Y_train)
            
            # regularization term gradients
            dWo += lmbd * output_weights
            dWh += lmbd * hidden_weights
            
            # update weights and biases
            output_weights -= eta * dWo
            output_bias -= eta * dBo
            hidden_weights -= eta * dWh
            hidden_bias -= eta * dBh

            #Loss[i] = -1 *np.sum( t *np.log(t +(1-Y_test)) * np.log(1-t))
        #"""
        test_accuracy[i][j] = accuracy_score(Y_train, predict(X_train))    
        if ((eta == 0.01) and (lmbd == 1)):
            print(predict(X_train))
        """
        test_accuracy[i][j] = accuracy_score(Y_test, predict(X_test))    
        if ((eta == 0.01) and (lmbd == 1)):
            print(predict(X_test))
        """

sns.set()
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()