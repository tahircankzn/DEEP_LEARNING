import os
import csv
from PIL import Image
import pandas as pd

# resimlerin bulunduğu klasörün yolu
folder_path = 'data2/'

# dönüştürülmüş verilerin kaydedileceği CSV dosyası adı
csv_file_name = 'data.csv'


#verilerim
df1 = pd.read_csv('epidural_N.csv')
df2 = pd.read_csv('intraparenchymal_N.csv')
df3 = pd.read_csv('intraventricular_N.csv')
df4 = pd.read_csv('none_N.csv')
df5 = pd.read_csv('subarachnoid_N.csv')
df6 = pd.read_csv('subdural_N.csv')


# CSV dosyasını oluştur
with open(csv_file_name, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # tüm resim dosyalarını klasörden al
    resim_dosya_adlari = os.listdir(folder_path)

    for dosya_adi in resim_dosya_adlari:
        # resim dosyasının yolunu oluştur
        
        resim_yolu = os.path.join(folder_path, dosya_adi)

        # resmi aç ve boyutunu yeniden boyutlandır
        resim = Image.open(resim_yolu)
        

        # resimdeki piksellerin gri tonlarını al
        gri_tonlar = list(resim.getdata())
        #gri_tonlar = [int(sum(piksel)/3) for piksel in pikseller]
        
        # resmin adını gri tonlar listesinin başına ekle
        if dosya_adi[:-4] in list(df1["epidural"]):
            #gri_tonlar.insert(0,[1,0,0,0,0,0])
            gri_tonlar.insert(0,1)
            gri_tonlar.insert(1,0)
            gri_tonlar.insert(2,0)
            gri_tonlar.insert(3,0)
            gri_tonlar.insert(4,0)
            gri_tonlar.insert(5,0)
            
            
        elif dosya_adi[:-4] in list(df2["intraparenchymal"]):
            gri_tonlar.insert(0,0)
            gri_tonlar.insert(1,1)
            gri_tonlar.insert(2,0)
            gri_tonlar.insert(3,0)
            gri_tonlar.insert(4,0)
            gri_tonlar.insert(5,0)
            
        elif dosya_adi[:-4] in list(df3["intraventricular"]):
            #gri_tonlar.insert(0,[0,0,1,0,0,0])
            gri_tonlar.insert(0,0)
            gri_tonlar.insert(1,0)
            gri_tonlar.insert(2,1)
            gri_tonlar.insert(3,0)
            gri_tonlar.insert(4,0)
            gri_tonlar.insert(5,0)
            
        elif dosya_adi[:-4] in list(df4["none"]):
            #gri_tonlar.insert(0,[0,0,0,0,0,0])
            gri_tonlar.insert(0,0)
            gri_tonlar.insert(1,0)
            gri_tonlar.insert(2,0)
            gri_tonlar.insert(3,0)
            gri_tonlar.insert(4,0)
            gri_tonlar.insert(5,0)
            
        elif dosya_adi[:-4] in list(df5["subarachnoid"]):
            #gri_tonlar.insert(0,[0,0,0,1,0,0])
            gri_tonlar.insert(0,0)
            gri_tonlar.insert(1,0)
            gri_tonlar.insert(2,0)
            gri_tonlar.insert(3,1)
            gri_tonlar.insert(4,0)
            gri_tonlar.insert(5,0)
            
        elif dosya_adi[:-4] in list(df6["subdural"]):
            #gri_tonlar.insert(0,[0,0,0,0,1,0])
            gri_tonlar.insert(0,0)
            gri_tonlar.insert(1,0)
            gri_tonlar.insert(2,0)
            gri_tonlar.insert(3,0)
            gri_tonlar.insert(4,1)
            gri_tonlar.insert(5,0)
            

        # gri tonları csv dosyasına yazdır
        csv_writer.writerow(gri_tonlar)
        


"""

epidural         [1,0,0,0,0,0]
intraparenchymal [0,1,0,0,0,0]
intraventricular [0,0,1,0,0,0]
subarachnoid     [0,0,0,1,0,0]
subdural         [0,0,0,0,1,0]
none             [0,0,0,0,0,0]

"""