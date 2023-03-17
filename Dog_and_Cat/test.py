#%%
import numpy as np


# 25000 resim var ve hepsi 28x28 = 2304 , datanın ilk sütünu olması gereken değerler
DATA = np.genfromtxt('train.csv', delimiter=',')

labels = DATA[:, 0]
labels = labels.astype(int)
y = np.zeros((25000))

for i in range(25000):
    
    y[i] = labels[i]

images = DATA[:,1:]
images = images/255 # normalization

#%%

images = images.transpose()

hn1 = 60  # Number of neurons in the first hidden layer
hn2 = 40  # Number of neurons in the second hidden layer

w12 = np.random.randn(hn1, 2304) * np.sqrt(2/2304)
w23 = np.random.randn(hn2,hn1) * np.sqrt(2/hn1)
w34 = np.random.randn(1,hn2) * np.sqrt(2/hn2)
b12 = np.random.randn(hn1,1)
b23 = np.random.randn(hn2,1)
b34 = np.random.randn(1)

#learning rate
eta = 0.0058


def elu(x): 

    if x>=0:
        return x
    else:
        return 0.2*(np.exp(x)-1)

def elup(x):
    if x >=0:
        return 1
    else:
        return 0.2 * np.exp(x)
    

epochs = 10


m = 10 #Minibatch size



    



i = 20000 
            
        # Feed forward
a1 = images[:,i:i+1]
z2 = w12.dot(a1) + b12
a2 = np.vectorize(elu)(z2)
z3 = w23.dot(a2) + b23
a3 = np.vectorize(elu)(z3)
z4 = w34.dot(a3) + b34
a4 = np.vectorize(elu)(z4) #Output vector

print(a4)


            