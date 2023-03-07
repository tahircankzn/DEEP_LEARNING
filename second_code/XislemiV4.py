import random
import time
import math
import numpy as np
import pandas as pd

# np.genfromtxt('veriler.csv', delimiter=',') # kayıtlı veriyi yükler


E = math.e # MATEMATİKSEL E'NİN SAYISAL DEĞERİ
learnig_value = 6 # ÖĞRENME KATSAYISI 5
step = 0.1 # ADIM SAYISI  0.0001
loop = 1000000 # BELİRLENEN TEKRAR SAYISI
liste_bölme = 100
counter = -1 # target değerleri eşleştirmek için bir sayaç

# VERİ SETİ


"""
DATA = [[],
        []]
for i in range(1000):
     DATA[0].append(1)
     DATA[1].append(2)

target = []
for i in range(1000):
     target.append(DATA[0][i] * DATA[1][i])
"""


#target = np.array(target)
target =   np.genfromtxt('target.csv', delimiter=',')                   


#DATA = np.array(DATA)  
DATA = np.genfromtxt('DATA.csv', delimiter=',')        



#w1 = 1#random.randint(1,2)
#w2 = 1#random.randint(1,2)
#w3 = 1#random.randint(1,2)    # BAŞLANGIÇ İÇİN RASGELE AĞIRLIK HESABI ATAMASI
#w4 = 1#random.randint(1,2)
#w5 = 1#random.randint(1,2)
#w6 = 1#random.randint(1,2)


#w_list_1 = np.array([[w1,w2],[w3,w4]])

w_list_1 =  np.genfromtxt('w_list_1.csv', delimiter=',')  

#w_list_2 = np.array([w5,w6])

w_list_2 =  np.genfromtxt('w_list_2.csv', delimiter=',')  

for i in range(10000):  

    def sigmoid(x):
            return 1 / (1+E**-x)

    def cost(x):

        global counter
        if counter < 999:
            counter+=1
    
        return (1/(2*1000)) * ( (target[counter] - x)**2 )

    # gizli katman çarpma toplama kısmı

    E0 = w_list_1.dot(DATA)


    # gizli katman sigmoid


    S1 = np.vectorize(sigmoid)(E0)


    # çıkış katman çarpma toplama kısmı

    E1 = w_list_2.dot(S1)
    
    if len(E1) != 1000:
        E1 = E1[0]
    # çıkış sigmoid

    S2 = np.vectorize(sigmoid)(E1)
    


    #cost fonksiyonu

    cost = np.vectorize(cost)(S2)
    cost_toplam = np.sum(cost)
    counter = -1

    print(cost_toplam)

    if cost_toplam > -0.1 and cost_toplam < 0.1:
        print("PASSED : ")
        print(w_list_1)
        print(w_list_2)
        pass

    
    
    # w5 için türev

    w5_türev = ( (-1/1000) * (np.array(target) - S2) ) * (S2 * (1 - S2)) * S1[0]

    # w6 için türev

    w6_türev = ( (-1/1000) * (np.array(target) - S2) ) * (S2 * (1 - S2)) * S1[1]


    

    
    # w3 için türev

    w3_türev = ( (-1/1000) * (np.array(target) - S2) ) * (S2 * (1 - S2)) * w_list_2[1] * (S1[1] * (1 - S1[1])) * DATA[0]

    # w4 için türev

    w4_türev = ( (-1/1000) * (np.array(target) - S2) ) * (S2 * (1 - S2)) * w_list_2[1] * (S1[1] * (1 - S1[1])) * DATA[1]


    # w1 için türev
    
    w1_türev = ( (-1/1000) * (np.array(target) - S2) ) * (S2 * (1 - S2)) * w_list_2[0] * (S1[0] * (1 - S1[0])) * DATA[0]
    
    # w2 için türev

    w2_türev = ( (-1/1000) * (np.array(target) - S2) ) * (S2 * (1 - S2)) * w_list_2[0] * (S1[0] * (1 - S1[0])) * DATA[1]




    w1_türev_toplamı = np.sum(w1_türev)
    w2_türev_toplamı = np.sum(w2_türev)
    w3_türev_toplamı = np.sum(w3_türev)
    w4_türev_toplamı = np.sum(w4_türev)
    w5_türev_toplamı = np.sum(w5_türev)
    w6_türev_toplamı = np.sum(w6_türev)



    w1_yeni = w_list_1[0][0] - 0.01 * w1_türev_toplamı
    w2_yeni = w_list_1[0][1] - 0.01 * w2_türev_toplamı
    w3_yeni = w_list_1[1][0] - 0.01 * w3_türev_toplamı
    w4_yeni = w_list_1[1][1] - 0.01 * w4_türev_toplamı
    w5_yeni = w_list_2[0] - 0.01 * w5_türev_toplamı
    w6_yeni = w_list_2[1] - 0.01 * w6_türev_toplamı






    w_list_1 = np.array([[w1_yeni,w2_yeni],
                        [w3_yeni,w4_yeni]])


    w_list_2 = np.array([w5_yeni,w6_yeni])

np.savetxt('DATA.csv', DATA, delimiter=',')
np.savetxt('w_list_1.csv', w_list_1, delimiter=',')
np.savetxt('w_list_2.csv', w_list_2, delimiter=',')
np.savetxt('target.csv', target, delimiter=',')

"""

np.dot(): İki matrisin çarpımını hesaplar.
np.transpose(): Bir matrisin transpozunu hesaplar.
np.linalg.inv(): Bir matrisin tersini hesaplar.
np.linalg.det(): Bir matrisin determinantını hesaplar.


"""
