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
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        # (BU kısımda dense katmanı kullanılıcak sonra)

        # transition layer 2 ->
        self.conv3 = nn.Conv2d(60,120,kernel_size=1)
        self.bn3 = nn.BatchNorm2d(120)
        self.conv3_drop = nn.Dropout2d()
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)

        # (BU kısımda dense katmanı kullanılıcak sonra)

        # transition layer 3 ->
        self.conv4 = nn.Conv2d(120,240,kernel_size=1)
        self.bn4 = nn.BatchNorm2d(240)
        self.conv4_drop = nn.Dropout2d()
        self.pool4 = nn.MaxPool2d(kernel_size=2,stride=2)

        # (BU kısımda dense katmanı kullanılıcak sonra)

        # classification layer->
        self.fc1 = nn.Linear(11760, 5000)
        self.fcDout = nn.Dropout()
        self.fc2 = nn.Linear(5000, 1000)
        
        self.fc3 = nn.Linear(1000, 500)

        self.fc4 = nn.Linear(500, 250)
        self.fc5 = nn.Linear(250, 5)
        



    def forward(self, x):

        

        # starting layer ->
        x = self.conv1(x)
        x = self.pool1(x)
        x = torch.relu(x)

        #y1 = x   # [64, 30, 55, 55]


        # transition layer 1 ->
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv2_drop(x)
        x = self.pool2(x)
        x = torch.relu(x)  # [64, 3840, 14, 14]

        

        # transition layer 2 ->
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv3_drop(x)
        x = self.pool3(x)
        x = torch.relu(x)
        


 

        # transition layer 3 ->
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.conv4_drop(x)
        #x = self.pool4(x)
        x = torch.relu(x)



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
        x = torch.relu(x)
        x = self.fcDout(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.fcDout(x)
        x = self.fc5(x)
        
        
        return torch.log_softmax(x, dim=1)

        

"""
__________________________________________________________________________________________________________________________________
opt.SGD
optimizer = opt.SGD(myModel.parameters(), lr=learning_rate, momentum=momentum)


                                        lr      batchSizeTrain   n_epochs    momentum

Test set: Accuracy: 2297/6000 (38%) -  0.0005 -      64       -    80      -   0.99     
Test set: Accuracy: 2000/6000 (33%) -  0.0005 -     256       -    10      -   0.99     
Test set: Accuracy: 2000/6000 (33%) -  0.001  -     256       -    10      -   0.99     
Test set: Accuracy: 2002/6000 (33%) -  0.01   -     256       -    10      -   0.99                      
Test set: Accuracy: 2507/6000 (42%) -  0.001  -      64       -    60      -   0.99      
Test set: Accuracy: 3259/6000 (54%) -  0.001  -      64       -    100     -   0.99     
Test set: Accuracy: 2479/6000 (41%) -  0.005  -      64       -    60      -   0.99     
Test set: Accuracy:        ?        -  0.008  -      64       -    60      -   0.99     
Test set: Accuracy:        ?        -  0.0001 -      64       -    60      -   0.99  
Test set: Accuracy:        ?        -  0.003  -      64       -    60      -   0.99   

__________________________________________________________________________________________________________________________________
torch.optim.Adam
optimizer = torch.optim.Adam(myModel.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002)

                                         lr     batchSizeTrain   n_epochs

Test set: Accuracy: 2262/6000 (38%)  - 0.001   -     64       -    10     
Test set: Accuracy: 2353/6000 (39%)  - 0.001   -     64       -    30  

"""
