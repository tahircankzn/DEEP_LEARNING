import torch
import torchvision
from torch.utils.data import Dataset, DataLoader # giriş versi bununla yapılır
import torch.optim as opt
import torch.nn.functional as F
import numpy as np
#from model import *
from densenet169 import *
from Dataloader2 import*

######
# çoklu target desteklenmiyor düzelt


#####




if __name__=="__main__":

    n_epochs = 10 # tüm veriyi 3 kez eğitim için kullanıcaz
    batchSizeTrain = 64 # veriyi 64 lü parçalara ayırıp ,her biri için güncelleme yapıp bir sonrakine ilerleme
   
    batchSizeTest = 1000
    learning_rate = 0.01
    
    momentum = 0.9
   
    device = "cpu"

    
    trainLoader = DataLoader(data(), batch_size=batchSizeTrain, shuffle=True)
    
    testLoader = DataLoader(data(), batch_size=batchSizeTest, shuffle=True)

    ##Eğitim ve test
    myModel=Network().to(device) # işlemcim ile vb çalıştırmak için

    print(myModel)
    # for p in myModel.parameters():
    #     print(p)
    optimizer = opt.SGD(myModel.parameters(), lr=learning_rate, momentum=momentum) # ağırlık güncellemesi
    #Eğitim
    myModel.train()
    for e in range(n_epochs):
        print(e+1)
        for batch,target in trainLoader:
            # print(i, (batch, target))
            optimizer.zero_grad() #Gradientleri sıfırla
            o=myModel.forward(batch.to(device))
            #loss=F.nll_loss(o,target.to(device))
            loss = F.cross_entropy(o, torch.argmax(target, dim=1).to(device))# cross entropy çok boyutlu targetlarda kullanılır
            loss.backward()
            optimizer.step() # ağırlık güncellemesi
            
    #Test
    myModel.eval() # evolate = değerlendir
    testLoss = 0
    correct = 0
    for batch,target in testLoader:
        # with torch.no_grad():
            o=myModel.forward(batch.to(device))
            # testLoss+=F.nll_loss(o, target.to(device))
            pred = o.data.max(1, keepdim=True)[1]
            #print(pred.shape)
            #print(target.shape)
            
            #correct += pred.eq(target.to(device).data.view_as(pred)).sum()
            correct += pred.eq(torch.argmax(target, dim=1).to(device).data.view_as(pred)).sum()
    testLoss /= len(testLoader.dataset)

  
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(testLoader.dataset),
        100. * correct / len(testLoader.dataset)))








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
fc3 çıktısı: torch.Size([64, 6])



"""