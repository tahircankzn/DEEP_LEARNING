import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class data(Dataset):
    def __init__(self):
        data = np.loadtxt('veri_kullanıma_hazır.csv', delimiter=',')
        self.veriler = data[:, 6:]
        self.hedefler = data[:, 0:5]

    def __len__(self):
        return len(self.veriler)

    def __getitem__(self, idx):
        #veri = self.veriler[idx].float()
        #hedef = self.hedefler[idx].float()
        
        veri = torch.tensor(self.veriler[idx].reshape(-1, 224, 224)).float()
        hedef = torch.tensor(self.hedefler[idx]).float()


        return veri, hedef


"""data = np.loadtxt('veri_kullanıma_hazır.csv', delimiter=',')
        # girdiler ve çıktıları ayırma
inputs = data[:, 6:]
outputs = data[:, 0:5]


veriler = inputs# 6000 x 1 x 224 x 224 boyutunda veri setiniz
hedefler = outputs# 6000 x 5 boyutunda hedefleriniz

veri_seti = VeriSeti(veriler, hedefler)
yigin_boyutu = 64 # yığın boyutunu burada belirleyebilirsiniz
veri_yukleyici = DataLoader(veri_seti, batch_size=yigin_boyutu, shuffle=True)
print(list(veri_yukleyici)[0])"""