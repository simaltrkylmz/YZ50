import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

#görev 1
print("""-----------------------
        Görev 1
-----------------------""")

#geçen haftayla aynı
words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)  # 27

#X ve Y veri setini kurma
block_size = 3 #bağlam penceresi: önceki 3 harf
X, Y = [], []


for w in words[:5]:
    context = [0] * block_size  #başlangıçta 3 tane nokta
    for ch in w + '.':  #kelimenin sonuna bitiş noktasını ekliyoruz.
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        print(''.join(itos[i] for i in context), '--->', itos[ix]) #ne olduğunu ekranda görmek için

        #bağlam penceresini bir adım kaydırıyoruz.
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)

print(f"X'in şekli: {X.shape} (örnek sayısı, 3 harf)")
print(f"Y'nin şekli: {Y.shape} (örnek sayısı)")

#embedding tablosu
# 27 harfin her birini 2 boyutlu (x, y) bir koordinatla başlatıyoruz
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)

#X'in içindeki bütün indekslerin 2 boyutlu karşılıklarını C'den çekiyoruz.
emb = C[X]

print(f"Embedding şekli: {emb.shape} (örnek sayısı, 3 harf, 2 boyut)")

#görev 2
print("""-----------------------
        Görev 2
-----------------------""")

#hidden layer (W1, b1): 3 harf * 2 boyut= 6 girdi. Karpathy videoda 100 nöron kullanıyor.
W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)

#output layer (W2, b2): 100 nörondan 27 (alfabe boyutu) ham skor üretiyoruz.
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)

#forward pass
# emb matrisi [32, 3, 2] boyutundaydı. matris çarpımında onu (6,100) boyutu ile çarpamayız
# view(-1, 6) diyerek 3 harfin 2'şer embedding'ini yan yana dizip [32, 6] boyutuna yeniden ayarlıyoruz.
# -1 sebebi: şu anda 32 yazdık ama ileride veri setindeki tüm kelimeleri kullanmaya karar verdiğimizde model çökerdi. -1 bilgisayara "ikinci boyut 6 olsun, ilk boyutu da toplam veri miktarına bakarak hesapla." diyor.
h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # gizli katman + tanh
logits = h @ W2 + b2                      # çıkış katmanı (ham skor olarak)
#sırayla matris boyutları: [32,6], bunu W1 ile çarpınca [32,100], bunu da W2 ile çarpınca [32.27]

#loss hesaplaması: elle ve pytorch ile
#elle hesaplama (geçen hafta)
counts = logits.exp()
probs = counts / counts.sum(dim=1, keepdim=True)
loss_manuel = -probs[torch.arange(32), Y].log().mean()

#pytorch'un fonksiyonu
loss_pytorch = F.cross_entropy(logits, Y)

print(f"Geçen Haftaki Manuel Loss: {loss_manuel.item()}")
print(f"F.cross_entropy Loss:      {loss_pytorch.item()}")


#görev 3
print("""-----------------------
        Görev 3
-----------------------""")

def build_dataset(words_list):
    block_size = 3
    X, Y = [], []
    for w in words_list:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


#veriyi bölme
random.seed(42)
words_shuffled = words.copy()
random.shuffle(words_shuffled) #sözlükteki isimler sıralı olduğu için onları karıştırıyoruz.
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
print(f"Veri setleri -> train: {Xtr.shape[0]}, dev: {Xdev.shape[0]}, test: {Xte.shape[0]}\n")


#parametreleri başlatma fonksiyonu (Sürekli sıfırlayacağımız için fonksiyona aldık)
def init_parameters():
    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((27, 2), generator=g)
    W1 = torch.randn((6, 100), generator=g)
    b1 = torch.randn(100, generator=g)
    W2 = torch.randn((100, 27), generator=g)
    b2 = torch.randn(27, generator=g)
    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True
    return parameters, C, W1, b1, W2, b2


#tek bir minibatch'i overfit etme
print("-Tek minibatch overfit testi-")
parameters, C, W1, b1, W2, b2 = init_parameters()
ix = torch.randint(0, Xtr.shape[0], (32,))  # Sadece 32 örnek çekiyoruz
Xb, Yb = Xtr[ix], Ytr[ix]

for i in range(1000):  #aynı veriyi 1000 kere eğitiyoruz
    emb = C[Xb]
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)
    #print(f"Loss: {loss.item()}")

    for p in parameters:
        p.grad = None
    loss.backward()
    for p in parameters:
        p.data += -0.1 * p.grad

print(f"Overfit Edilmiş Loss Değeri (Sıfıra çok yakın olmalı): {loss.item():.4f}\n")

#veri setinin tamamını eğitmeden önce rastgele içinden 32 kelimeyi çekiyoruz ve bunları eğitiyoruz.


#learning rate'e karar verme
print("-Learning rate tarama-")
parameters, C, W1, b1, W2, b2 = init_parameters()  #ağırlıkları sıfırladık

lre = torch.linspace(-3, 0, 1000)
lrs = 10 ** lre  # 0.001'den 1'e kadar 1000 farklı learning rate deneyeceğiz
lri = []
lossi = []

for i in range(1000):
    ix = torch.randint(0, Xtr.shape[0], (32,))
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    for p in parameters: p.grad = None
    loss.backward()

    lr = lrs[i]
    for p in parameters: p.data += -lr * p.grad

    lri.append(lre[i])
    lossi.append(loss.item())

print("Tarama sonunda en optimum değerin 10^-1 (0.1) civarlarında olduğu sonucuna ulaştık\n")
plt.plot(lri, lossi)
plt.show()


#asıl eğitim
print("-Asıl eğitim-")
parameters, C, W1, b1, W2, b2 = init_parameters()  #gerçek eğitim için ağırlıkları son kez sıfırlıyoruz

for i in range(10000):
    ix = torch.randint(0, Xtr.shape[0], (32,)) #tüm kelimeleri aynı anda sormak yerine eğitim setinden rastgele 32 tane çekiyoruz

    #forward pass
    emb = C[Xtr[ix]] #32 kelimenin harflerini koordinata çeviriyoruz
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1) #gizli katman (100 nöron)
    logits = h @ W2 + b2 #alfabedeki 27 harf için ham skorlar
    loss = F.cross_entropy(logits, Ytr[ix]) #gerçek cevaplarla karşılaştırıp hatayı buluyoruz


    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    #ağırlıkları güncelleme
    #ilk 5000 adımda hızlı gidiyor, sonrasında hedefe yaklaşınca ince ayar için adımlarını küçültüyoruz (lr decay- hız düşürme)
    lr = 0.1 if i < 5000 else 0.01
    for p in parameters: p.data += -lr * p.grad

print(f"Eğitim sonu Train Loss: {loss.item():.4f}")

#eğittiğimiz modeli hiç görmediği dev seti ile test edip asıl loss'u buluyoruz.
emb = C[Xdev]
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
logits = h @ W2 + b2
dev_loss = F.cross_entropy(logits, Ydev)
print(f"Dev Seti Üzerindeki Gerçek Hata (Dev Loss): {dev_loss.item():.4f}")






