dosya = open("w.txt","r")
veri = dosya.read()
liste = list(veri.split(" "))
w1 = float(liste[0])
w2 = float(liste[1])
bias = float(liste[2])
dosya.close()


number1 = 12
number2 = 12


# nöron 1
nöron1_çıkış = number1 * w1 + number2 * w2 + bias 
    
    
print(nöron1_çıkış)
    


    




    


     







