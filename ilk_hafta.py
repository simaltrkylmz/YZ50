
#görev 1

import math

inputs= [0.7, 1.45, -0.85] #girdiler
weights=[0.4, 0.6, 0.3] #ağırlıklar
bias= 0.68 #biasımız
summation=0
for i, input in enumerate (inputs): #bize hem indexi hem de inputs listesindeki input değerlerini veriyor.
    weight= weights[i] #aynı indexteki weight değerleri
    summation+= input*weight #input ve weight çarpımlarının toplamı
summation+=bias #döngü sonunda bias değerini hesaplıyor
#sigmoid fonksiyonuyla çarpma
output= 1/(1+math.exp(-summation))
print(output)

#görev2 birden fazla nöronlu katman
# input değerlerim aynı kalıyor ama bir sonraki aşamada artık 3 nöronum olacak, o yüzden weight ve bias değerlerim her nöron için değişmeli
inputs= [0.7, 1.45, -0.85] #girdiler
weights=[[0.4, 0.6, 0.3],  #birinci nöron için weight değerleri
         [0.1, 0.5, -0.6], #ikinci nöron için weight değerleri
         [0.2, -0.7, 0.8]] #üçüncü nöron için weight değerleri
biases=[0.68, -0.45, 0.56]
outputs_layer= [] #artık üç outputum olacak, o yüzden outputlarımı bir listeye ekleyeceğim

for i, weight in enumerate(weights): #weights iç içe listesinin iç listeleri (yani nöronların tek tek weightleri)
    bias= biases[i]  #bu nöronların bias değerleri
    summation=0
    for j, input in enumerate(inputs): #iç içe for döngüsü. yukarıda yazılanın aynısı, her nöron için summation hesaplıyor
        summation+= input*weight[j]
    summation+=bias
    outputs_layer.append(1/(1+math.exp(-summation))) #output değerlerini hesaplayıp outputs_layer listesine ekliyor
print(outputs_layer)



#görev3
#loss fonksiyonu (mean squared error kullanılarak)
targets=[1.0,1.0,0.0] #ulaşmak istediklerimiz
def loss_function(targets,outputs_layer): #loss fonksiyonunu hesaplayan fonksiyonumuz
    loss= 0
    for target, output in zip(targets, outputs_layer): #iki listedeki elemanları gezmemizi sağlayan for döngüsü
        loss+= ((target - output)**2) #ikisinin farkının karesi
    loss/= len(targets) #en son çıkan toplam değerini eleman sayısına bölme
    return loss
print(loss_function(targets,outputs_layer))






