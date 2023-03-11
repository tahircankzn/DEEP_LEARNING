import random
import time


w1 = random.randint(1,5)
w2 = random.randint(1,5)
bias = random.randint(1,5)


for i in range(10000):
    

    number1 = 12
    number2 = 12

    result = number1*number2

    # nöron 1
    nöron1_çıkış = number1 * w1 + number2 * w2 + bias
    
    #cost_function = (1/2)*((result - nöron1_çıkış)**2)

    def derivative(number):
        return ((nöron1_çıkış - result)*-1)*number

    yakınlık= result - nöron1_çıkış

    if  yakınlık == 0:  #if yakınlık < 0.01 and yakınlık > -0.01:   # if yakınlık == 0:
        #print(f"PASSED  {number1} * {number2} = {result} , nöron çıkışı : {nöron1_çıkış} , yakınlık : {yakınlık} ,cost : {yakınlık} , w1 : {w1} - w2 : {w2}")
        print(f"yakınlık : {yakınlık} , w1 : {w1} - w2 : {w2}")
        dosya = open("w.txt","w")
        dosya.write(f"{w1} {w2}")
        dosya.close()
        break
    else:
        w1 = w1 + derivative(number1) * 0.001 
        w2 = w2 + derivative(number2) * 0.001
        bias = bias + ((nöron1_çıkış - result)*-1)
        #print(f"nöron çıkışı : {nöron1_çıkış} , yakınlık : {yakınlık} , w1 : {w1} - w2 : {w2}")
    
print(f"nöron çıkışı : {nöron1_çıkış} , yakınlık : {yakınlık} , w1 : {w1} - w2 : {w2} , bias : {bias}")
dosya = open("w.txt","w")
dosya.write(f"{w1} {w2} {bias}")
dosya.close()
    
    
    
    

    


     







