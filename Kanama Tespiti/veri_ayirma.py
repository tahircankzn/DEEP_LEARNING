
# stage_2_train dosyasındakileri kanama türüne göre ayırma ve kaydetme


import csv
from random import shuffle

hold_none = []
hold_1 = []
hold_2 = []
hold_3 = []
hold_4 = []
hold_5 = []

with open('stage_2_train.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    liste = list(csvreader)

    #print(liste[1][0])

    #print(liste[:8])


    for i in range(1,len(liste)-4,6):

        """
        if liste[i][0][:12] == "ID_f9ffeb014": # deneme
            print(liste[i])
            print(liste[i+1])
            print(liste[i+2])
            print(liste[i+3])
            print(liste[i+4])
            
        """
        

        hold_index = []
        
        if liste[i][1] == "0":
            hold_index.append("0")
            
        else:
            hold_index.append("1")
        

        if liste[i+1][1] == "0":
            hold_index.append("0")
        else:
            hold_index.append("1")


        if liste[i+2][1] == "0":
            hold_index.append("0")
        else:
            hold_index.append("1")


        if liste[i+3][1] == "0":
            hold_index.append("0")
        else:
            hold_index.append("1")

        if liste[i+4][1] == "0":
            hold_index.append("0")
        else:
            hold_index.append("1")

            
        
        

        
        if hold_index == ['0', '0', '0', '0', '0'] and len(hold_none) <= 1000:
            hold_none.append(liste[i][0][:12])
            
        if hold_index == ['1', '0', '0', '0', '0'] and len(hold_1) <= 1000:
            hold_1.append(liste[i][0][:12])
            
        if hold_index == ['0', '1', '0', '0', '0'] and len(hold_2) <= 1000:
            hold_2.append(liste[i][0][:12])
            
        if hold_index == ['0', '0', '1', '0', '0'] and len(hold_3) <= 1000:
            hold_3.append(liste[i][0][:12])
            
        if hold_index == ['0', '0', '0', '1', '0'] and len(hold_4) <= 1000:
            hold_4.append(liste[i][0][:12])
            
        if hold_index == ['0', '0', '0', '0', '1'] and len(hold_5) <= 1000:
            hold_5.append(liste[i][0][:12])
            
"""
print(hold_none[:5],['0', '0', '0', '0', '0'],len(hold_none))
print(hold_1[:5],['1', '0', '0', '0', '0'],len(hold_1))
print(hold_2[:5],['0', '1', '0', '0', '0'],len(hold_2))
print(hold_3[:5],['0', '0', '1', '0', '0'],len(hold_3))
print(hold_4[:5],['0', '0', '0', '1', '0'],len(hold_4))
print(hold_5[:5],['0', '0', '0', '0', '1'],len(hold_5))
"""




def create(name,liste):

    csv_file_name = f'{name}.csv'

    # CSV dosyasını oluştur
    with open(csv_file_name, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        for i in liste:
            csv_writer.writerow([i])


create("noneY2",hold_none)
create("epiduralY2",hold_1)
create("intraparenchymalY2",hold_2)
create("intraventricularY2",hold_3)
create("subarachnoidY2",hold_4)
create("subduralY2",hold_5)

"""
hold_none.extend(hold_1)
hold_none.extend(hold_2)
hold_none.extend(hold_3)
hold_none.extend(hold_4)
hold_none.extend(hold_5)


shuffle(hold_none)
shuffle(hold_none)
shuffle(hold_none)

create("data",hold_none)
"""
