from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def loss(Y,ao):
    return (1.0/n)*np.sum((Y - ao)**2)

#derivative of loss
def d_L(Y,ao):
    print(Y.shape, ao.shape)
    return (2.0/n)*np.sum(Y - ao)
    
def SGD(X,y,lmbd, eta):
    for i in range(nbatch):
        random_index = M*np.random.randint(nbatch)
        xi = X[random_index:random_index+M]
        yi = y[random_index:random_index+M]
        dWo, dBo, dWh, dBh = backpropagation(xi, yi)
        
        # regularization term gradients
        dWo += lmbd * Wo
        dWh += lmbd * Wh

        
    return Wo, Bo, Wh, Bh

def d_activ():
    return 1    

def activation(x):
    #sigmoid
    #f = 1/(1 + np.exp(-x))
    
    #Linear
    f = x
    
    #2. order
    #f= x**2

    return f


def feed_forward(X):
        
    # weighted sum of inputs to the hidden layer
    zh = X @ Wh + Bh
    # activation in the hidden layer
    ah = activation(zh)

    # weighted sum of inputs to the output layer
    zo = ah @ Wo + Bo
    ao = activation(zo)
    print(ao.shape)
    
    # for backpropagation need activations in hidden and output layers
    return ah, ao

def backpropagation(X, Y):
    ah, ao = feed_forward(X)  

    # error in the output layer
    eo = d_L(Y,ao)
    print("Her",eo.shape)
    # error in the hidden layer
    eh = eo @ Wo.T #d_a
    
    
    # gradients for the output layer
   #dWo = d_L() @ d_acitv() @ ah
    dWo = ah.T @ eo
    dBo = np.sum(eo, axis=0)
    
    # gradient for the hidden layer
    dWh = X.T @ eh
    dBh = np.sum(eh, axis=0)
    

    return dWo, dBo, dWh, dBh

# we obtain a prediction by taking the class with the highest likelihood
def predict(X):
    ah, ao = feed_forward(X)
    return np.argmax(ao, axis=1)


# ensure the same random numbers appear every time
np.random.seed(0)
n = 5
deg = 2
eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)
epochs= 100
lim = 0.001
itter_limit = 10000

M = 2   #size of each minibatch

# data:
x = 2*np.random.rand(n,1) 
Y = 3*x + 4 #+ noise

# Design matrix
poly = PolynomialFeatures(deg)
X = poly.fit_transform(x)

nbatch = int(len(X)/M) #number of minibatches


# Defining the neural network
n_inputs, n_features = X.shape # # (10,3) 10 sett med (x0,x1,x2)
n_hidden_neurons = 5 # 
n_categories = 3 # 3 output (beta0,beta1,beta2)
n_features = 3 # x0, x1 og x2


# we make the weights normally distributed using numpy.random.randn
# weights and bias in the hidden layer
Wh = np.random.randn(n_features, n_hidden_neurons)
Bh = np.zeros(n_hidden_neurons) + 0.01

# weights and bias in the output layer
Wo = np.random.randn(n_hidden_neurons, n_categories)
dWo = np.random.randn(n_hidden_neurons, n_categories)
Bo = np.zeros(n_categories) + 0.01

test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

#loop over etas and lambdas
for i, eta in enumerate(eta_vals):
   for j, lmbd in enumerate(lmbd_vals):
        # loop over epochs
        epoch = 0
        while np.linalg.norm(dWo) > lim :
            epoch +=1
            # calculate gradient
            dWo, dBo, dWh, dBh = SGD(X,Y,lmbd,eta)
            if epoch > itter_limit:
                print("break SGD")
                break
            # update weights and 
            Wo -= eta * dWo
            Bo -= eta * dBo
            Wh -= eta * dWh
            Bh -= eta * dBh
            
            #calculate gradient for whole dataset to use as a stop criteria 
            dWo, dBo_t, dWh_t, dBh_t  = backpropagation(X, Y)

        if np.any(np.isnan(dWo)):
            print(eta, lmbd)

        test_accuracy[i][j] = accuracy_score(predict(X),Y)    
        print(predict(X))


sns.set()
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()