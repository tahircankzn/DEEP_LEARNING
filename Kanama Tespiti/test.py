import torch
import torchvision
from torch.utils.data import Dataset, DataLoader # giriş versi bununla yapılır
import torch.optim as opt
import torch.nn.functional as F
import numpy as np
#from model import *
from densenet169_2 import *   # densenet169
from testDataLoader import*


if __name__=="__main__":

    n_epochs = 20 # tüm veriyi 3 kez eğitim için kullanıcaz
    batchSizeTrain = 64 # veriyi 64 lü parçalara ayırıp ,her biri için güncelleme yapıp bir sonrakine ilerleme
   
    batchSizeTest = 625
    learning_rate = 0.001 # 0.001     %42
    
    momentum = 0.99 # 0.9
   
    device = "cpu"

    
    
    
    testLoader = DataLoader(data(), batch_size=batchSizeTest, shuffle=True)

    ##Eğitim ve test
    myModel=Network().to(device) # işlemcim ile vb çalıştırmak için

    print(myModel)
    # for p in myModel.parameters():
    #     print(p)



    optimizer = opt.SGD(myModel.parameters(), lr=learning_rate, momentum=momentum) # ağırlık güncellemesi
    # deneme optimizer
    #optimizer = torch.optim.Adam(myModel.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002)
    #optimizer = opt.RMSprop(myModel.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0.0001)
    myModel.load_state_dict(torch.load('model_weights.pth'))
    
    
    
    #Test
    myModel.eval() # evolate = değerlendir
    testLoss = 0
    correct = 0
    for batch,target in testLoader:
        
            o=myModel.forward(batch.to(device))
            
            pred = o.data.max(1, keepdim=True)[1]
           
            correct += pred.eq(torch.argmax(target, dim=1).to(device).data.view_as(pred)).sum()
    testLoss /= len(testLoader.dataset)

  
    print('\nTest set: Accuracy: {}/{} ({:.0f}%) - lr : {}\n'.format(
        correct, len(testLoader.dataset),
        100. * correct / len(testLoader.dataset),learning_rate))
    
    








"""

conv1 çıktısı: torch.Size([64, 30, 112, 112])
pool1 çıktısı: torch.Size([64, 30, 55, 55])
conv2 çıktısı: torch.Size([64, 60, 28, 28])
bn2 çıktısı: torch.Size([64, 60, 28, 28])
conv2_drop çıktısı: torch.Size([64, 60, 28, 28])
pool2 çıktısı: torch.Size([64, 60, 14, 14])
conv3 çıktısı: torch.Size([64, 120, 14, 14])
bn3 çıktısı: torch.Size([64, 120, 14, 14])
conv3_drop çıktısı: torch.Size([64, 120, 14, 14])
pool3 çıktısı: torch.Size([64, 120, 7, 7])
conv4 çıktısı: torch.Size([64, 240, 7, 7])
bn4 çıktısı: torch.Size([64, 240, 7, 7])
conv4_drop çıktısı: torch.Size([64, 240, 7, 7])
pool4 çıktısı: torch.Size([64, 240, 3, 3])
fc1 çıktısı: torch.Size([64, 500])
fcDout çıktısı: torch.Size([64, 500])
fc2 çıktısı: torch.Size([64, 50])
fc3 çıktısı: torch.Size([64, 5])



"""