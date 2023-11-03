from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Loss fuction
def loss(X):
    #MSE
    l = (1.0/n)*np.sum((Y - predict(X))**2, axis=0) 
    return l

#derivative of loss
def dL(X,Y):
    dl = np.sum(Y - predict(X), axis=0)    
    return dl
   
def activation(code, z):
    if code == 'sigmoid': 
        f = 1/(1 + np.exp(-z))
    elif code =='relu':
        if z >= 0: f = z
        else: f = 0
        #zero = np.zeros(len(z))
        #f = np.max([z,zero], axis=0)   
    elif code == 'softmax':
        f = np.exp(z)/np.sum(np.exp(z),axis=0)
    return f

def dA(a):
    #sigmoid
    da = a * (1-a)
    
    #relu
    #da = 1 
    
    #Softmax
    #da =  a * (1-a)
    return da

    
   

def feed_forward(X):
        
    # weighted sum of inputs to the hidden layer
    zh = X @ Wh + Bh
    # activation in the hidden layer
    ah = activation('sigmoid',zh)

    # weighted sum of inputs to the output layer
    zo = ah @ Wo + Bo
    ao = activation('softmax',zo)
    
    # for backpropagation need activations in hidden and output layers
    return ah, ao


def backpropagation(X, Y):
    ah, ao = feed_forward(X)  

    # error in the output layer
    #eo = dL(X,Y) # Derivative of dL/dao
    eo = ao - Y
    
    # error in the hidden layer
    eh = eo @ Wo.T * dA(ah)
    
    # gradients for the output layer
    #dWo = dL() @ da() @ ah
    #dWo = ah.T @ np.reshape(eo,(10,1))
    dWo = ah.T @ eo
    dBo = np.sum(eo, axis=0)
   
    # gradient for the hidden layer
    dWh = X.T @ eh
    dBh = np.sum(eh, axis=0)
    

    return dWo, dBo, dWh, dBh


def SGD(X,y,lmbd):
    for i in range(nbatch):
        random_index = M*np.random.randint(nbatch)
        xi = X[random_index:random_index+M]
        yi = y[random_index:random_index+M]
        dWo, dBo, dWh, dBh = backpropagation(xi, yi)
        
        # regularization term gradients
        dWo += lmbd * Wo
        dWh += lmbd * Wh
        
    return dWo, dBo, dWh, dBh

def predict(X):
    ah, ao = feed_forward(X)
    return ao#np.argmax(ao, axis=1)

def accuracy_score_numpy(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)




####################################################
#Setting up variables
np.random.seed(0)
n = 5       # nuber of datapoint
epochs= 100 # Number of epochs, itterations throug NN
lim = 0.001 # Fradient limit for gradient decent
itter_limit = 10000 # Itterations limit it gradient does not converge
eta_vals = np.logspace(-5, 1, 7) #Learn rate values
lmbd_vals = np.logspace(-5, 1, 7) #Penalty values
M = 10   #size of each minibatch in SGD
 

# data:
noise = 1.5*np.random.rand(n,1)
X = 2*np.random.rand(n,1) 
Y = 3*X**2 + 4 + noise
#plt.plot(X,Y,'*')

#number of minibatches for SGD
nbatch = int(len(X)/M) 


# Defining the neural network
n_inputs, n_features = X.shape # rows and columns of input
n_hidden_neurons = 1#len(X) # Chose what gives best results
n_categories = 1#len(X) # outputs


# we make the weights normally distributed using numpy.random.randn
# weights and bias in the hidden layer
Wh = np.random.randn(n_features, n_hidden_neurons)
Bh = np.zeros(n_hidden_neurons) + 0.01

# weights and bias in the output layer
Wo = np.random.randn(n_hidden_neurons, n_categories)
dWo = np.random.randn(n_hidden_neurons, n_categories)
Bo = np.zeros(n_categories) + 0.01

#print(Y)
#print(predict(X))
dWo_t, dBo_t, dWh_t, dBh_t  = backpropagation(X, Y)

test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

#loop over learnrates and penalties
for i, eta in enumerate(eta_vals):
   for j, lmbd in enumerate(lmbd_vals):
        # loop over epochs, through NN
        epoch = 0
        while np.linalg.norm(dWo_t) > lim :
            epoch +=1
            # calculate gradient
            #not using SGD
            dWo, dBo, dWh, dBh = backpropagation(X, Y)
            # regularization term gradients
            dWo += lmbd * Wo
            dWh += lmbd * Wh
            
            #Using SGD
            #dWo, dBo, dWh, dBh = SGD(X,Y,lmbd)
            
            if epoch > itter_limit:
                print("break SGD")
                break
            # update weights and 
            Wo -= eta * dWo
            Bo -= eta * dBo
            Wh -= eta * dWh
            Bh -= eta * dBh
            
            #calculate gradient for whole dataset to use as a stop criteria 
            dWo_t, dBo_t, dWh_t, dBh_t  = backpropagation(X, Y)

        if np.any(np.isnan(dWo)):
            print(eta, lmbd)

        test_accuracy[i][j] = np.sum(Y - predict(X) / len(Y))   
        print(predict(X))


sns.set()
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()