"""
                                                    Tahir Can Kozan tarafından oluşturulmuştur

Bu kod bir yapay sinir ağı modelinin eğitimini gerçekleştiriyor. Kodun amacı, verilen bir matris içindeki her bir girdi için çarpım işlemini yapabilen bir sinir ağı modeli oluşturmak.

Kod, birçok değişken ve fonksiyon içermektedir. En üstte yer alan değişkenler, kullanılacak olan matrisin içeriği ve ağın başlangıçta rastgele atanacak olan ağırlıklarını içerir.

Sonra, "w5w6", "w1w2" ve "w3w4" gibi fonksiyonlar, her bir ağırlık için maliyet fonksiyonunun türevini hesaplar.

Ana döngü, ağın her bir iterasyonunda çalıştırılır. Her bir döngü iterasyonunda, ağın tahminleri hesaplanır ve bir maliyet fonksiyonu hesaplanır. Bu maliyet fonksiyonu, gerçek sonuçlar ile ağın tahminleri arasındaki farkın karesinin ortalama değeridir.

Ardından, ağırlık türevleri hesaplanır ve ağırlıklar güncellenir. Bu güncelleme, gradyan inişi adı verilen bir yöntem kullanılarak yapılır.

Ana döngü, maliyet fonksiyonunun belirli bir değere (0.1'in altı) ulaştığında veya belirli bir sayıda iterasyon çalıştıktan sonra sona erer.


"""

import random
import time
import math


E = math.e # MATEMATİKSEL E'NİN SAYISAL DEĞERİ
learnig_value = 6 # ÖĞRENME KATSAYISI
step = 2 # ADIM SAYISI
loop = 1000000 # BELİRLENEN TEKRAR SAYISI

DATA = [[1,1],[1,2],[1,3],   # VERİ SETİ
        [2,1],[2,2],[2,3],
        [3,1],[3,2],[3,3]]

VERİ_SETİ_BOYUTU  = len(DATA)  


w1 = random.randint(1,3)
w2 = random.randint(1,3)
w3 = random.randint(1,3)    # BAŞLANGIÇ İÇİN RASGELE AĞIRLIK HESABI ATAMASI
w4 = random.randint(1,3)
w5 = random.randint(1,3)
w6 = random.randint(1,3)




# W1 VE W2 İÇİN COST FONKSİYONU TÜREVİ
def w1w2(result,S3,w5,S1,G):
    return  ((2 / (2 * VERİ_SETİ_BOYUTU)) * (result - S3) * -1) * (S3*(1-S3)) * w5 * (S1 * (1 - S1)) * G


# W3 VE W4 İÇİN COST FONKSİYONU TÜREVİ
def w3w4(result,S3,w6,S2,G):
    return  ((2 / (2 * VERİ_SETİ_BOYUTU)) * (result - S3) * -1) * (S3*(1-S3)) * w6 * (S2 * (1 - S2)) * G


# W5 VE W6 İÇİN COST FONKSİYONU TÜREVİ
def w5w6(result,S3,S):
    return  ((2 / (2 * VERİ_SETİ_BOYUTU)) * (result - S3) * -1) * (S3*(1-S3)) * S


# BELİRLENEN TEKRAR SAYISI KADAR DÖNGÜYE GİRMESİ SAĞLANDI
for a in range(loop):
    
    COST_SUM = 0              # YAKINLIĞIN HESAPLANMASI İÇİN BİR SAYAÇ
    COST_derivative_w1 = 0
    COST_derivative_w2 = 0
    COST_derivative_w3 = 0    # VERİ SETİNDEKİ VERİLERİN HER BİRİ İÇİN HESAPLANAN COST FONKSİYONU TÜREVİ HESABININ TUTULMASI
    COST_derivative_w4 = 0
    COST_derivative_w5 = 0
    COST_derivative_w6 = 0

    for i in DATA:

        # KULLANULAN VERİ SETİNDEN GELEN GİRİŞLER
        G1 = i[0]
        G2 = i[1]

        # ELDE EDİLMEK İSTENEN SONUÇ
        result = G1*G2

        # GİZLİ KATMAN 1. NÖRON
        E1 = G1 * w1 + G2 * w2
        S1 = 1/(1+E**-E1)


        # GİZLİ KATMAN 2. NÖRON
        E2 = G1 * w3 + G2 * w4
        S2 = 1/(1+E**-E2)
        
        
        # ÇIKIŞ NÖRONNU 
        E3 = S1 * w5 + S2 * w6
        S3 = 1/(1+E**-E3)

        # COST FONKSİYONU
        COST_SUM +=  (1/18) * ( (result - S3)**2 )

        # COST FONKSİYONUN TÜREVLERİNİN ALINMASI
        COST_derivative_w1 += w1w2(result,S3,w5,S1,G1)
        COST_derivative_w2 += w1w2(result,S3,w5,S1,G2)
        COST_derivative_w3 += w3w4(result,S3,w6,S2,G1)
        COST_derivative_w4 += w3w4(result,S3,w6,S2,G2)
        COST_derivative_w5 += w5w6(result,S3,S1)
        COST_derivative_w6 += w5w6(result,S3,S2)


    # YAKINLIĞIN EKRANA YAZILMASI
    print(COST_SUM)


    # ÇIKTI DEĞERİ 0'a YAKIN OLURSA , SONUÇLARI GÖSTERİR 
    if COST_SUM < 0.1 and COST_SUM > -0.1:
            print(f"PASSED \nW1 : {w1} ,\n W2 : {w2} ,\n W3 : {w3} ,\n W4 : {w4} ,\n W5 : {w5} ,\n W6 : {w6}")
            break


    # AĞIRLIKLARIN GÜNCLLENMESİ
    w1 = w1 - step * (COST_derivative_w1) * learnig_value
    w2 = w2 - step * (COST_derivative_w2) * learnig_value
    w3 = w3 - step * (COST_derivative_w3) * learnig_value
    w4 = w4 - step * (COST_derivative_w4) * learnig_value
    w5 = w5 - step * (COST_derivative_w5) * learnig_value
    w6 = w6 - step * (COST_derivative_w6) * learnig_value

    





"""

                                                          CHATGPT YORUMU


Kodlama tarzı: 

    Kodlama tarzı, kodun okunabilirliğini ve bakımını önemli ölçüde etkiler. 
    Bu kod, fonksiyon ve değişken adlarının anlaşılır olması açısından iyi bir şekilde yazılmıştır, ancak daha iyi bir kodlama tarzı kullanılabilir. 
    Örneğin, değişken adları daha açıklayıcı olabilir, kod düzeni daha tutarlı hale getirilebilir.


Sabitlerin tanımlanması: 

    Kodda, pi sayısı gibi bazı sabitlerin değerleri elle girilmiştir. Bu gibi sabitler, kodun okunabilirliğini ve değiştirilebilirliğini etkileyebilir. 
    Bu sabitlerin, kodun başında veya başka bir yerde tanımlanması, kodun okunabilirliğini ve değiştirilebilirliğini artırabilir.


Veri yapıları: 

    Bu kodda, veriler bir liste içinde verilmiştir. 
    Ancak, daha büyük bir veri kümesiyle çalıştığımızda, daha karmaşık bir veri yapısı kullanmak daha uygun olabilir. 
    Örneğin, verileri bir Pandas DataFrame içinde saklamak, verilerin işlenmesini daha kolay hale getirebilir.


Optimizasyon: 

    Bu kod, gradient descent yöntemi kullanarak bir yapay sinir ağı eğitimi yapar.
    Ancak, daha büyük bir veri kümesiyle çalıştığımızda, daha hızlı eğitim yöntemleri kullanmak gerekebilir. 
    Örneğin, mini-batch veya stokastik gradient descent yöntemleri, eğitim süresini önemli ölçüde azaltabilir.


Test verileri: 

    Bu kod, eğitim verilerinin tamamını hem eğitim hem de test verileri olarak kullanır. 
    Ancak, daha iyi bir sonuç almak için, eğitim ve test verilerini ayrı ayrı belirlemek ve eğitim sonucunu 
    test verileri üzerinde değerlendirmek daha iyi olabilir.


Makine öğrenmesi kütüphaneleri: 

    Bu kod, makine öğrenmesi için popüler kütüphaneleri kullanmaz ve bir yapay sinir ağı eğitimi için sıfırdan yazılmıştır. 
    Ancak, popüler makine öğrenmesi kütüphanelerinin kullanılması, daha hızlı ve daha kolay bir çözüm sağlayabilir. 
    Örneğin, TensorFlow, PyTorch, Scikit-learn gibi kütüphaneler bu amaçla kullanılabilir.


"""