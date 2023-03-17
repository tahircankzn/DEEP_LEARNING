#%%
import numpy as np
import time


#start = time.time()

# 25000 resim var ve hepsi 28x28 = 2304 , datanın ilk sütünu olması gereken değerler
DATA = np.genfromtxt('train.csv', delimiter=',')
#%%
labels = DATA[:, 0]
labels = labels.astype(int)
y = np.zeros((25000))

for i in range(25000):
    
    y[i] = labels[i]

images = DATA[:,1:]
images = images/255 # normalization

images = images.transpose()

hn1 = 60  # Number of neurons in the first hidden layer
hn2 = 40  # Number of neurons in the second hidden layer
"""
w12 = np.genfromtxt('wtwo.csv', delimiter=',')#np.random.randn(hn1, 2304) * np.sqrt(2/2304)
w23 = np.genfromtxt('wthree.csv', delimiter=',')#np.random.randn(hn2,hn1) * np.sqrt(2/hn1)
w34 = np.genfromtxt('wfour.csv', delimiter=',')#np.random.randn(10,hn2) * np.sqrt(2/hn2)
b12 = np.genfromtxt('btwo.csv', delimiter=',')#np.random.randn(hn1,1)
b23 = np.genfromtxt('bthree.csv', delimiter=',')#np.random.randn(hn2,1)
b34 = np.genfromtxt('bfour.csv', delimiter=',')#np.random.randn(1)
"""
#"""
w12 = np.random.randn(hn1, 2304) * np.sqrt(2/2304)
w23 = np.random.randn(hn2,hn1) * np.sqrt(2/hn1)
w34 = np.random.randn(1,hn2) * np.sqrt(2/hn2)
b12 = np.random.randn(hn1,1)
b23 = np.random.randn(hn2,1)
b34 = np.random.randn(1)
#"""

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

#print(time.time()-start)
for k in range(epochs): #Outer epoch loop
    batches = 0
    print(k+1)
    for j in range(int(25000/m)):
        error4 = np.zeros((1))
        error3 = np.zeros((hn2,1))
        error2 = np.zeros((hn1,1))
        errortot4 = np.zeros((1,1))
        errortot3 = np.zeros((hn2,1))
        errortot2 = np.zeros((hn1,1))
        grad4 = np.zeros((w34.shape))
        grad3 = np.zeros((w23.shape))
        grad2 = np.zeros((w12.shape))


        for i in range(batches,batches+m):
            
            # Feed forward
            a1 = images[:,i:i+1]
            
            z2 = w12.dot(a1) + b12
            a2 = np.vectorize(elu)(z2)
            
            z3 = w23.dot(a2) + b23
            a3 = np.vectorize(elu)(z3)
            z4 = w34.dot(a3) + b34
            a4 = np.vectorize(elu)(z4) #Output vector


            error4 = (a4-y[i]) * np.vectorize(elup)(z4)
            error3 =  ((w34.transpose()).dot(error4))* np.vectorize(elup)(z3)
            error2 =  ((w23.transpose()).dot(error3))* np.vectorize(elup)(z2)

            errortot4 = errortot4 + error4
            errortot3 = errortot3 + error3
            errortot2 = errortot2 + error2
            grad4 = grad4 + error4.dot(a3.transpose())
            grad3 = grad3 + error3.dot(a2.transpose())
            grad2 = grad2 + error2.dot(a1.transpose())
            
            # Gradient descent
        
        w34 = w34 - (eta*grad4)/m
        w23 = w23 - eta/m*grad3
        w12 = w12 - eta/m*grad2
        b34 = b34 - eta/m*errortot4
        b23 = b23 - eta/m*errortot3
        b12 = b12 - eta/m*errortot2
        
        batches = batches + m
        
    
    idx = np.random.permutation(images.shape[1])
    images = images[:, idx]
    y = y[idx]

np.savetxt('wfour.csv', w34, delimiter=',')
np.savetxt('wthree.csv', w23, delimiter=',')      
np.savetxt('wtwo.csv', w12, delimiter=',')
np.savetxt('bfour.csv', b34, delimiter=',')  
np.savetxt('bthree.csv', b23, delimiter=',')
np.savetxt('btwo.csv', b12, delimiter=',')  
# %%
