from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Loss fuction
def loss(ao,Y):
    if c == 1: 
        #Cross Entrypy
        L = accuracy_score(ao,Y)
        #not implemented properly
    else:
        #MSE
        L = (1.0/n)*np.sum((Y - ao)**2, axis=0) 

    return L



#derivative of loss
    if c == 1: 
        #derivative of Cross Entrypy
        dl = ao-Y
    else:
        #derivative of MSE
        dl = (2/len(Y))*(ao - Y)
    return dl


   
def activation(code, z):
    if code == 'sigmoid': 
        f = 1/(1 + np.exp(-z))
        
    elif code =='relu':
        zero = np.zeros(z.shape)
        f = np.max([z,zero], axis=0)  
        
    elif code == 'leakrelu':
        f = np.where(z > 0, z, a*z)
   
    elif code == 'elu':
        f = np.where(z > 0, z, a*(np.exp(z)-1)) 
        
    elif code == 'softmax':
        f = np.exp(z)/np.sum(np.exp(z),axis=0)
        
    elif code == 'linear':
        f = z    
    return f



def dA(code, x):
    if code == 'sigmoid': 
        df = x*(1-x)
        
    elif code =='relu':
        df = 1 * (x > 0)
          
    elif code == 'leakrelu':
        df = np.where(x > 0, 1, a)  
         
    elif code == 'elu':
        df = np.where(x > 0, 1, activation('elu',x)*a)  
        
    elif code == 'softmax':
        df = x*(1-x) #if i=j) 
        #df = -x_i *x_j # if i != j
        # not properly implemented
        
    elif code == 'linear':
        df = 1
        
    return df

    
   

def feed_forward(X):
        
    # weighted sum of inputs to the hidden layer
    zh = X @ Wh + Bh
    # activation in the hidden layer
    ah = activation(a_hidden,zh)

    # weighted sum of inputs to the output layer
    zo = ah @ Wo + Bo
    ao = activation(a_out,zo)

    # for backpropagation need activations in hidden and output layers
    return ah, ao


def backpropagation(X, Y):
    ah, ao = feed_forward(X)  

    # error in the output layer
    eo = dL(ao,Y) # derivative of the loss function
    
    # error in the hidden layer
    #eh = dL @ Wo.T * da
    eh = eo @ Wo.T * dA(a_hidden,ah)
    
    # gradients for the output layer
    #dWo = dL() @ da() @ ah
    dWo = ah.T @ eo
    dBo = np.sum(eo, axis=0)
   
    # gradient for the hidden layer
    dWh = X.T @ eh
    dBh = np.sum(eh, axis=0)
    #print(eh)

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
        
    if c == 1: pred = np.argmax(ao, axis=1)
    else: pred = ao
            
    return pred





####################################################
#Setting up variables
np.random.seed(0)
n = 100      # nuber of datapoint
epochs= 100 # Number of epochs, itterations throug NN
lim = 0.00001 # Gradient limit for gradient decent
itter_limit = 10000 # Itterations limit it gradient does not converge
eta_vals = np.logspace(-5, -1, 5) #Learn rate values
lmbd_vals = np.logspace(-5, 0, 4) #Penalty values
Nh = 5
M = 10   #size of each minibatch in SGD
c=0 # not classification

#possible activation functions: 
#sigmoid, relu, leakrelu, elu, softmax, linear 
a_hidden = "relu"  # choise of activation function
a_out = "linear"

a=0.001 #Alpha parameter in elu and leaky relu

# Data:
##Two-domesional Exponetial
#noise = 0.1*np.random.rand(n,2)
#X = 2*np.random.rand(n,2) 
#Y = 3*np.exp(X) + 4 + noise

##Linear funtion
noise = 0.1*np.random.rand(n,1)
X = 2*np.random.rand(n,1)
Y = 3*X + 4 + noise

##Classification:
#c = 1
#X = np.array([ [0, 0], [0, 1], [1, 0],[1, 1]],dtype=np.float64)
#yAND = np.array( [ 0, 1 ,1, 1])
#Y = np.reshape(yAND,(len(yAND),1))

#plt.plot(X,Y,'*')

#number of minibatches for SGD
nbatch = int(len(X)/M) 

# Defining the neural network
n_inputs, n_features = X.shape # rows and columns of input
n_hidden_neurons = Nh # Chose what gives best results
n_categories = np.shape(X)[1] # outputs




test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
#print("True: ")
#print(Y)
#print("  ")


#loop over learn-rates and penalties
for i, eta in enumerate(eta_vals):
    print(eta)
    for j, lmbd in enumerate(lmbd_vals):      
         # loop over epochs, through NN
         
         # weights and bias in the hidden layer
         Wh = np.random.randn(n_features, n_hidden_neurons)
         Bh = np.zeros(n_hidden_neurons) + 0.01
    
         # weights and bias in the output layer
         Wo = np.random.randn(n_hidden_neurons, n_categories)
         Bo = np.zeros(n_categories) + 0.01
         dWo_t = np.random.randn(n_hidden_neurons, n_categories)
        
         epoch = 0
         while np.linalg.norm(dWo_t) > lim :
             epoch +=1
             # calculate gradient
             dWo, dBo, dWh, dBh = backpropagation(X, Y)
             #if epoch ==2: print(dWo)
             # regularization term gradients
             dWo += lmbd * Wo
             dWh += lmbd * Wh
    
             #Using SGD
             #dWo, dBo, dWh, dBh = SGD(X,Y,lmbd)
             
             if epoch > itter_limit:
                 print("break ")
                 break
             
             # update weights and 
             Wo -= eta * dWo
             Bo -= eta * dBo
             Wh -= eta * dWh
             Bh -= eta * dBh
             
             #calculate gradient for whole dataset to use as a stop criteria 
             dWo_t, dBo_t, dWh_t, dBh_t  = backpropagation(X, Y)
    
             test_accuracy[i][j] = loss(predict(X),Y) 

        #print("number of itterations: ",epoch)
        #print(predict(X))



sns.set()
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

