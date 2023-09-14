
import numpy as np
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4


def create_X(x, y, n ):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X


np.random.seed(2168)
# Making meshgrid of datapoints and compute Franke's function
d = 3
N = 1000
nlambdas = 10
Lambda = np.logspace(-4, 0, nlambdas)
MSEPredictRidge = np.zeros(nlambdas)
MSETrainRidge = np.zeros(nlambdas)
MSEPredictLasso = np.zeros(nlambdas)
MSETrainLasso = np.zeros(nlambdas)
#R2_Ridge = np.zeros(nlambdas)

x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
z = FrankeFunction(x, y)+ 0.1*np.random.randn(N)
X = create_X(x, y, d)    
#X = np.identity(N)

# split in training and test data
X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)


#scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
""" no scaling on y_test"""


#Calculate beta and prediction
RegOLS = skl.LinearRegression()
RegOLS.fit(X_train_scaled,z_train)
zpredictOLS = RegOLS.predict(X_test_scaled)
MSE_OLS_Train= mean_squared_error(RegOLS.predict(X_train_scaled), z_train)
MSE_OLS_test = mean_squared_error(RegOLS.predict(X_test_scaled), z_test)
R2_OLS_test = RegOLS.score(X_test_scaled,z_test)

for i in range(nlambdas):

    RegRidge = skl.Ridge(Lambda[i])
    RegRidge.fit(X_train_scaled,z_train)
    ypredictRidge = RegRidge.predict(X_test_scaled)
    MSEPredictRidge[i] = mean_squared_error(RegRidge.predict(X_test_scaled), z_test)
    MSETrainRidge[i] = mean_squared_error(RegRidge.predict(X_train_scaled), z_train)
    #R2_Ridge[i] = RegRidge.score(X_test_scaled,z_test)

    RegLasso = skl.Lasso(Lambda[i])
    RegLasso.fit(X_train_scaled,z_train)
    zpredictLasso = RegLasso.predict(X_test_scaled)
    MSEPredictLasso[i]= mean_squared_error(RegLasso.predict(X_test_scaled), z_test)
    MSETrainLasso[i]= mean_squared_error(RegLasso.predict(X_train_scaled), z_train)
    #R2_Lasso_test = RegLasso.score(X_test_scaled,z_test)




print("  ")
print("MSE OLS: {:.8f}".format(MSE_OLS_Train))
print("R2 score OLS: {:.8f}".format(R2_OLS_test))

MSE_OLS = np.ones(nlambdas)*MSE_OLS_test
plt.figure()
plt.plot(np.log10(Lambda), MSE_OLS, '--',label= "OLS test")
plt.plot(np.log10(Lambda), MSEPredictRidge,'r',  label = 'Ridge test')
plt.plot(np.log10(Lambda), MSETrainRidge, 'g', label = 'Ridge train')
#plt.plot(np.log10(Lambda), MSEPredictLasso,'k',  label = 'Lasso test')
#plt.plot(np.log10(Lambda), MSETrainLasso, 'y', label = 'Lasso train')
plt.grid()
plt.xlabel('log10(lambda)')
plt.ylabel('MSE')
plt.title('Polynomial degree: ' +str(d))
plt.legend()
plt.show()














