
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




