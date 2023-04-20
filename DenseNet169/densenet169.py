import torch.nn as nn
import torch
import torch.nn.functional as F

## 64,1,224,224
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1,30,kernel_size=7,stride=2,padding=3) # 64 , 30 , 112 , 112
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        
        # (BU kısmda dense katmanı kullanılıcak sonra)

        # transition layer 1 ->
        self.conv2 = nn.Conv2d(30,60,kernel_size=1,stride=2) # 64 , 30 , 56 , 56
        self.bn2 = nn.BatchNorm2d(60)
        self.conv2_drop = nn.Dropout2d()
        self.pool2 = nn.AvgPool2d(kernel_size=2,stride=2)

        # (BU kısımda dense katmanı kullanılıcak sonra)

        # transition layer 2 ->
        self.conv3 = nn.Conv2d(60,120,kernel_size=1)
        self.bn3 = nn.BatchNorm2d(120)
        self.conv3_drop = nn.Dropout2d()
        self.pool3 = nn.AvgPool2d(kernel_size=2,stride=2)

        # (BU kısımda dense katmanı kullanılıcak sonra)

        # transition layer 3 ->
        self.conv4 = nn.Conv2d(120,240,kernel_size=1)
        self.bn4 = nn.BatchNorm2d(240)
        self.conv4_drop = nn.Dropout2d()
        self.pool4 = nn.AvgPool2d(kernel_size=2,stride=2)

        # (BU kısımda dense katmanı kullanılıcak sonra)

        # classification layer->
        self.fc1 = nn.Linear(11760, 500)
        self.fcDout = nn.Dropout()
        self.fc2 = nn.Linear(500, 50)

        self.fc3 = nn.Linear(50, 6)

    """
    def denseblok_DENEME(self, conv, pool, size):
        self.conv = nn.Conv3d(3, size, kernel_size=3, stride=2)
        self.pool = nn.Conv3d(size, size, kernel_size=1)
        conv_output = conv(self.conv)
        pool_output = pool(self.pool)
        output = conv_output + pool_output
        return output
    """   
    
    def denseblok_2(self,x,r,size):
            for _ in range(r):
                y = nn.Conv2d(size,size,kernel_size=1)(x)
                y = nn.Conv2d(size,size,kernel_size=3)(y)
                x = x + y
            return x
    


    def forward(self, x):

        

        # starting layer ->
        x = self.conv1(x)
        x = self.pool1(x)
        x = torch.relu(x)

        # dense layer ->
        #x = self.denseblok_2(x,6,30)

        # transition layer 1 ->
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv2_drop(x)
        x = self.pool2(x)
        x = torch.relu(x)

        # dense layer ->
        #x = self.denseblok_2(x,6,60)

        # transition layer 2 ->
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv3_drop(x)
        x = self.pool3(x)
        x = torch.relu(x)

        # dense layer ->
        #x = self.denseblok_2(x,6,120)

        # transition layer 3 ->
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.conv4_drop(x)
        #x = self.pool4(x)
        x = torch.relu(x)

        # dense layer ->
        #x = self.denseblok_2(x,6,240)



        #print(x.__len__())
        #print(x[0].__len__())
        #print(x.size())
        #print(x[0][0].__len__())



        # Convolüsyon çıkışını vektör haline getir
        x = x.view(-1, 11760) #   64 , 240x7x7 = 11760             64 , 240x3x3 = 2160

        # Lineer katmanların çıkışını hesapla
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fcDout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fcDout(x)
        x = self.fc3(x)

        return torch.log_softmax(x, dim=1)

        

