
#görev 1

import math
import matplotlib.pyplot as plt
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
targets=[1.0,0.5,0.0] #ulaşmak istediklerimiz (görev 4teki grafiğin parabolik çıkması için 2. değeri 0.5 yaptım. (sınır değeri olmaması için))
def loss_function(targets,outputs_layer): #loss fonksiyonunu hesaplayan fonksiyonumuz
    loss= 0
    for target, output in zip(targets, outputs_layer): #iki listedeki elemanları gezmemizi sağlayan for döngüsü
        loss+= ((target - output)**2) #ikisinin farkının karesi
    loss/= len(targets) #en son çıkan toplam değerini eleman sayısına bölme
    return loss
print(loss_function(targets,outputs_layer))


#görev4
def forward_pass(inputs, weights, biases): #görev 1deki forward pass'i fonksiyon şeklinde yazıyoruz çünkü weighti değiştirdiğimizde tekrar kullanmamız gerekecek
    outputs_layer = []
    for weight,bias in zip(weights,biases): #görev 1de yazdığım for döngüsü (farklı olarak zip ile iki listeyi aynı anda aramayı kullandım)
        summation=0
        for x,w in zip (inputs, weight):
            summation+= x*w
        summation+=bias
        outputs_layer.append(1/(1+math.exp(-summation)))
    return outputs_layer

original_weight = weights[1][1] #değiştireceğimiz weight değerinin orijinal değerini saklıyoruz

test_weights = [] # x ekseni için boş liste
losses = [] # y ekseni için boş liste

for i in range (-50,51): #-5'ten 5'e kadar 0.1 artışla weight'i arttıran döngü
    test_weight=i/10.0
    weights[1][1]= test_weight
    current_guess= forward_pass(inputs, weights, biases)
    current_loss=loss_function(targets,current_guess)
    print(f"Weight is: {test_weight} Loss is: {current_loss}")
    test_weights.append(test_weight)
    losses.append(current_loss)

weights[1][1]= original_weight #döngüden sonra weightin orijinal değerini geri getiriyoruz

#grafiği çizme
plt.plot(test_weights, losses, label='Loss Eğrisi', color='purple')
plt.title("Ağırlık Değişimine Göre Loss Değişimi")
plt.xlabel("Ağırlık (weights[2][2])")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.show()
