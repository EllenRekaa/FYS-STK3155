from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

#Loss fuction
def loss(ao,Y):
    if c == 1: 
        #Cross Entrypy
        L = accuracy_score(ao,Y)
        #not implemented properly
    else:
        #MSE
        L = (1.0/len(Y))*np.sum((Y - ao)**2, axis=0) 

    return L



#derivative of loss
def dL(ao,Y):
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

    return dWo, dBo, dWh, dBh


def predict(X):
    ah, ao = feed_forward(X)
        
    if c == 1: pred = np.argmax(ao, axis=1)
    else: pred = ao
            
    return pred




####################################################

#Setting up variables
np.random.seed(0)

#possible activation functions: 
#sigmoid, relu, leakrelu, elu, softmax, linear 
a_hidden = "sigmoid"  # choise of activation function
a_out = "softmax"

#n = 500      # nuber of datapoint in regression
Nh = 100 # Number of neurons in hidden layer
epochs= 100 # Number of epochs, itterations throug NN
lim = 10**-4 # Gradient limit
itter_limit = 10000 # Itterations limit if gradient does not converge
eta_vals = np.logspace(-5, -3, 7) #Learn rate values
lmbd_vals = np.logspace(-5, -1, 7) #Penalty values
c=0 # not classification
a=0.001 #Alpha parameter in elu and leaky relu






#########################################

# Data:
"""
##Two-domesional Exponetial
noise = 0.1*np.random.rand(n,2)
X = 2*np.random.rand(n,2) 
Y = 3*np.exp(X) + 4 + noise


##Linear funtion
noise = 0.1*np.random.rand(n,1)
X = 2*np.random.rand(n,1)
Y = 3*np.exp(X) + 4 + noise


##Classification:
c = 1
#X = np.array([ [0, 0], [0, 1], [1, 0],[1, 1]],dtype=np.float64)
X = np.array([ [0, 0], [0, 1], [1, 0],[1, 1], [0, 0], [0, 1], 
              [1, 0],[1, 1], [0, 0], [0, 1], [1, 0],[1, 1]],
             dtype=np.float64)
#yAND = np.array( [ 0, 1 ,1, 1])
yAND = np.array( [ 0, 1 ,1, 1, 0, 1 ,1, 1, 0, 1 ,1, 1])
Y = np.reshape(yAND,(len(yAND),1))
"""

##Brestcanser Data
#Download breast cancer dataset
c=1
cancer = load_breast_cancer()      
#Feature matrix of 569 rows (samples) and 30 columns (parameters)
Data  = cancer.data 
# 0 for benign and 1 for malignant  
Y = cancer.target       
labels  = cancer.feature_names[0:30]
print(labels)

X = Data[:,0:9]
Y = np.reshape(Y, (len(Y),1))

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)


#####################################
# Defining the neural network
n_inputs, n_features = X_train.shape # rows and columns of input
n_hidden_neurons = Nh # Chose what gives best results
n_categories = np.shape(X_train)[1] # outputs



########################################
train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

#loop over learn-rates and penalties
for i, eta in enumerate(eta_vals):
    
    print("Learn Rate: ",eta)
    for j, lmbd in enumerate(lmbd_vals):      
         
         # Reset weights and bias in the hidden layer
         Wh = np.random.randn(n_features, n_hidden_neurons)
         Bh = np.zeros(n_hidden_neurons) + 0.01
    
         # Resets weights and bias in the output layer
         Wo = np.random.randn(n_hidden_neurons, n_categories)
         Bo = np.zeros(n_categories) + 0.01
         dWo_t = np.random.randn(n_hidden_neurons, n_categories)
         
         # loop over epochs, through NN, 
         # stops when gradient is smaller than lim
         epoch = 0
         while np.linalg.norm(dWo_t) > lim :
             epoch +=1
             # calculate gradient
             dWo, dBo, dWh, dBh = backpropagation(X_train, Y_train)
            
             # regularization term gradients
             dWo += lmbd * Wo
             dWh += lmbd * Wh
             
             if epoch > itter_limit:
                 print("break ")
                 break
              
                 
             # Update weights with learn rate
             Wo -= eta * dWo
             Bo -= eta * dBo
             Wh -= eta * dWh
             Bh -= eta * dBh
             
             # Select best values for plotting (Hard coded from previous run)
             if (eta == 10**-2) and (lmbd == 10**-2):
                 Wo_best = Wo
                 Bo_best = Bo
                 Wh_best = Wh
                 Bh_best = Bh
             
             # Calculate gradient for whole Train dataset 
             # to use as a stop criteria 
             dWo_t, dBo_t, dWh_t, dBh_t  = backpropagation(X_train, Y_train)
             
             #Store loss value
             train_accuracy[i][j] = loss(predict(X_train),Y_train) 
             test_accuracy[i][j] = loss(predict(X_test),Y_test)


#Print Results

lmbd_labels= np.round(np.log10(lmbd_vals),2)
eta_labels= np.round(np.log10(eta_vals),2)

sns.set()
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_accuracy, annot=True, 
            fmt='.3g', ax=ax, cmap="viridis",
            xticklabels= lmbd_labels, yticklabels=eta_labels)
ax.set_title("Train Loss")
ax.set_ylabel(" log( $\eta$ )")
ax.set_xlabel("log( $\lambda$ )")
plt.show()

sns.set()
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, 
            annot=True, ax=ax, cmap="viridis",
            xticklabels= lmbd_labels, yticklabels=eta_labels)
ax.set_title("Test Loss")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()


"""
Wo = Wo_best
Bo = Bo_best
Wh = Wh_best
Bh = Bh_best
y_pred = predict(X)

fig = plt.figure()
plt.plot(X,y_pred,'o',label="Predicted")
plt.plot(X,Y,'.',label="Real")
plt.grid()
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Real and predicted values")
plt.legend()
plt.show()

"""


