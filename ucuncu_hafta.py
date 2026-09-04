import urllib.request
import torch
import matplotlib.pyplot as plt

#görev 1: bigram modelini anlamak ve veri setindeki kelimelerden harf ikililerini elde edip sayaç oluşturmak
url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
urllib.request.urlretrieve(url, "names.txt") #karpathy'nin isim listesini indirme

words = open('names.txt', 'r').read().splitlines()

chars= sorted(set(''.join(words))) #alfabeyi elde etme. sorted liste döndürdüğü için ekstradan list yazmadım.
print(chars)
stoi={} #string to integer için başta boş dictionary
stoi['.']=0 #0. indexi nokta yapma
for sayi, harf in enumerate(chars): #a'yı 1, b'yi 2... şeklinde her harfe sayı değeri atama
    stoi[harf]=sayi+1
print(stoi)

itos={} #integer to string için başta boş dictionary
for harf,sayi in stoi.items(): #yukarıdakinin tam tersi, 0'a nokta, 1'e a... atama
    itos[sayi]=harf
print(itos)

b={}
for w in (words):
    chs= ['.'] + list(w) + ['.']
    for ch1,ch2 in zip(chs,chs[1:]): #ikili şekilde gruplama: i. ve i+1. olarak
        bigram=(ch1,ch2)
        if bigram in b: #eğer sözlüğün içindeyse 1 arttırma
            b[bigram]+=1
        else:
            b[bigram]=1 #değilse değerini 1 yapma
sirali_b = sorted(b.items(), key=lambda kv: -kv[1]) #sözlükteki en çok olan 10 tanesini yazdırma
print(sirali_b[:10])

#pytorch ile yazdırma

# 27x27 boyutunda içi 0'larla dolu matris
N = torch.zeros((27, 27), dtype=torch.int32)
for w in (words):
    chs= ['.'] + list(w) + ['.'] #üstteki döngünün benzeri
    for ch1,ch2 in zip(chs,chs[1:]):
        N[stoi[ch1],stoi[ch2]]+=1 #tekrar eden ikililerde kutucuğun değerini 1 arttırıyoruz

#görselleştirme
"""plt.figure(figsize=(16, 16)) # 16x16 inçlik dev bir tuval aç
plt.imshow(N, cmap='Blues')  # matrisi Mavi (Blues) tonlarıyla renklendir

# tablonun içindeki 27x27 = 729 kutucuğun hepsini tek tek geziyoruz
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j] #kutunun içine yazılacak harf ikilisi
        # harfleri kutunun üstüne yazma
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        # sayıları kutunun altına yazma
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off') #kenardaki ekran çizgilerini gizleme
plt.show()      #tabloyu ekrana getirme"""

#görev 2
#satır satır olasılıklara çevirme
P=N.float() / N.sum(dim=1, keepdim=True) #keepdim yapmazsak sağdan başlayarak sütunlara böler ve sessiz ama çok büyük hata alırız.
print(P.sum(dim=1))

#yeni isimler örnekleme

g = torch.Generator().manual_seed(2147483647) #videodaki değeri girdim ama çıktılar aynı olmadı. ai'a sordum, pytorch versiyonuyla ilgilidir dedi.
#generator'da aynı g'yi döngüde tekrar tekrar verdiğim için her seferinde aynı çıktıyı aldım.
for i in range(5):
    out=[]
    ix=0
    while True:
        # multinomial bize olasılıklarla orantılı çıktılar verir. yüksek olasılıklı olanlar daha sık çıkar.
        ix=torch.multinomial(P[ix], num_samples=1, replacement=True, generator=g).item() #bir tane sayı veriyor. o anda aldığını da torbaya geri koyuyor (replacement=True ile).
        out.append(itos[ix])
        if ix==0: #. karakterine gelince döngüden çıkıyor. çünkü o son karakterimiz.
            break
    print(''.join(out))

"""test = torch.tensor([0.7, 0.2, 0.1]) #multinomial'ı daha iyi anlamak için örnek kod
for _ in range(10):
    print(torch.multinomial(test, num_samples=1).item())"""
