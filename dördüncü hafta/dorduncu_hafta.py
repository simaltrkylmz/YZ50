import torch
import torch.nn.functional as F

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












