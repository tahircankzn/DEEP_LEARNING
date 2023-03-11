import random

w1 = random.randint(1,5)
w2 = random.randint(1,5)
bias = random.randint(1,5)
learning_rate = 0.001 

for i in range(10000):
    

    number1 = 200
    number2 = 200

    result = number1*number2

    # nöron 1
    nöron1_çıkış = number1 * w1 + number2 * w2 + bias

    def derivative(number):
        return ((nöron1_çıkış - result)*-1)*number

    yakınlık= result - nöron1_çıkış

    w1 = w1 + derivative(number1) * learning_rate
    w2 = w2 + derivative(number2) * learning_rate
    bias = bias + ((nöron1_çıkış - result)*-1)
    learning_rate = learning_rate - 0.001
        
    
print(f"nöron çıkışı : {nöron1_çıkış} , yakınlık : {yakınlık} , w1 : {w1} - w2 : {w2} , bias : {bias}")
dosya = open("w.txt","w")
dosya.write(f"{w1} {w2} {bias}")
dosya.close()
