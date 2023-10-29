from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def feed_forward(X):
        
    # weighted sum of inputs to the hidden layer
    zh = X.dot(Wh) + Bh
    # activation in the hidden layer
    ah = sigmoid(zh)

    # weighted sum of inputs to the output layer
    zo = ah.dot(Wo) + Bo
    ao = sigmoid(zo)

    
    # for backpropagation need activations in hidden and output layers
    return ah, ao

def backpropagation(X, Y):
    ah, ao = feed_forward(X)  

    # error in the output layer
    eo = ao - Y

    # error in the hidden layer
    eh = eo.dot(Wo.T) * ah * (1 - ah)

    
    # gradients for the output layer
    dWo = ah.T.dot(eo)
    dBo = np.sum(eo, axis=0)
    
    # gradient for the hidden layer
    dWh = X.T.dot(eh)
    dBh = np.sum(eh, axis=0)
    
    t = predict(X)

    return t, ao, dWo, dBo, dWh, dBh

# we obtain a prediction by taking the class with the highest likelihood
def predict(X):
    ah, ao = feed_forward(X)
    return np.argmax(ao, axis=1)


# ensure the same random numbers appear every time
np.random.seed(0)
e = 0.00001
l = 0.00001
eta_vals = np.logspace(-5, -2, 4)
lmbd_vals = np.logspace(-5, 1, 7)
epochs= 100
lim = 0.01

# data:
# Design matrix
X = np.array([ [0, 0], [0, 1], [1, 0],[1, 1]],dtype=np.float64)
#X = np.array([[0, 0]],dtype=np.float64)

# The XOR gate
yXOR = np.array( [ 0, 1 ,1, 0])
# The OR gate
yOR = np.array( [ 0, 1 ,1, 1])
# The AND gate
yAND = np.array( [ 0, 0 ,0, 1])
    
y = yAND
Y = np.reshape(y,(len(y),1))

# Defining the neural network
n_inputs, n_features = X.shape # (4,2) 4 sett med (x1,x2)
n_hidden_neurons = 2 # 
n_categories = 2 # 2 output (0,1)
n_features = 2 # x1 og x2


# we make the weights normally distributed using numpy.random.randn
# weights and bias in the hidden layer
Wh = np.random.randn(n_features, n_hidden_neurons)
Bh = np.zeros(n_hidden_neurons) + 0.01

# weights and bias in the output layer
Wo = np.random.randn(n_hidden_neurons, n_categories)
Bo = np.zeros(n_categories) + 0.01

print(y)
print(predict(X))
print("New accuracy on training data: " + str(accuracy_score(predict(X), Y)))


#Loss = np.zeros(epochs)
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

#loop over etas and lambdas
for i, eta in enumerate(eta_vals):
   for j, lmbd in enumerate(lmbd_vals):
        # loop over epochs
        for k in range(epochs):
            # calculate gradients
            t,P, dWo, dBo, dWh, dBh = backpropagation(X, Y)
            #if np.any(np.isnan(dWo)):
            #    print(eta, lmbd, k)
            
            # regularization term gradients
            dWo += lmbd * Wo
            dWh += lmbd * Wh
            
            # update weights and biases
            Wo -= eta * dWo
            Bo -= eta * dBo
            Wh -= eta * dWh
            Bh -= eta * dBh
        
            #t= predict(X)
            #-(1.0 / target.size) * np.sum(target * np.log(X + 10e-10))
            #Loss[k] = -1*np.sum( t *np.log(t +(1-Y_test)) * np.log(1-t))
            #Loss[k] = -1*np.sum( t - np.log(1-t))
        if np.any(np.isnan(dWo)):
            print(eta, lmbd)

        test_accuracy[i][j] = accuracy_score(Y, t)#predict(X))    
        #if ((eta == e) and (lmbd == l)):
        #    print(P)
        #    print(predict(X))
        print(predict(X))


sns.set()
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

