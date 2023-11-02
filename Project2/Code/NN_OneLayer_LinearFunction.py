from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
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
    eh = eo.dot(Wo.T) * ah #* (1 - ah)

    
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
    ypred = X.dot(ao.T)
    diff= abs(ypred-Y)
    minst = np.argmin(diff, axis=1)

    print(minst)
    print(ypred[:,4])
    return ypred[:,4]


# ensure the same random numbers appear every time
np.random.seed(0)
e = 0.00001
l = 0.00001
eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)
epochs= 1000


# data:
# Making meshgrid of datapoints and compute Franke's function
#noise = 0.1*np.random.normal(0,1,10)
#x = np.random.uniform(0, 1, 10)
x = 2*np.random.rand(5,1) 
y = 3*x + 4 #+ noise
#y = np.sort(np.random.uniform(0, 1, 100))
#z = FrankeFunction(x, y) + noise    
    
# Design matrix
poly = PolynomialFeatures(2)
X = poly.fit_transform(x)

Y = y#np.reshape(y,(len(y),1))

# Defining the neural network
n_inputs, n_features = X.shape # (10,3) 10 sett med (x0,x1,x2)
n_hidden_neurons = 5 # 
n_categories = 3 # 3 output (beta0,beta1,beta2)
n_features = 3 # x0, x1 og x2


# we make the weights normally distributed using numpy.random.randn
# weights and bias in the hidden layer
Wh = np.random.randn(n_features, n_hidden_neurons)
Bh = np.zeros(n_hidden_neurons) + 0.01

# weights and bias in the output layer
Wo = np.random.randn(n_hidden_neurons, n_categories)
Bo = np.zeros(n_categories) + 0.01

print("y-values ",Y)
diff = predict(X)
print("New accuracy on training data: " + str(accuracy_score(Y, predict(X))))


#Loss = np.zeros(epochs)
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

#loop over etas and lambdas
for i, eta in enumerate(eta_vals):
   for j, lmbd in enumerate(lmbd_vals):
        # loop over epochs
        for k in range(epochs):
            # calculate gradents
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

