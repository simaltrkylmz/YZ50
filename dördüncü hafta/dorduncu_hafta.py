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
print(X.data)
print(Y.data)

#embedding tablosu
# 27 harfin her birini 2 boyutlu (x, y) bir koordinatla başlatıyoruz
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)

#C 27 harfin her biri için rastgele başlatılmış 2 boyutlu vektör tutan tablo

#X'in içindeki her sayı için (yani her bağlamdaki her 3 harf için), C'nin o harfe denk gelen satırını çekiyoruz
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
#sırayla matris boyutları: [32,6], bunu W1 ile çarpınca [32,100], bunu da W2 ile çarpınca [32,27]

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

for i in range(200):  #aynı veriyi 200 kere eğitiyoruz
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

for i in range(50000):
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
    #ilk 25000 adımda hızlı gidiyor, sonrasında hedefe yaklaşınca ince ayar için adımlarını küçültüyoruz (lr decay- hız düşürme)
    lr = 0.1 if i < 25000 else 0.01
    for p in parameters: p.data += -lr * p.grad

print(f"Eğitim sonu Train Loss: {loss.item():.4f}")

#eğittiğimiz modeli hiç görmediği dev seti ile test edip asıl loss'u buluyoruz.
emb = C[Xdev]
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
logits = h @ W2 + b2
dev_loss = F.cross_entropy(logits, Ydev)
print(f"Dev Seti Üzerindeki Gerçek Hata (Dev Loss): {dev_loss.item():.4f}")


#görev 4
print("""-----------------------
        Görev 4
-----------------------""")

#-2 boyutlu embedding haritasını çizme-
#görev 3'teki C matrisini kullanıyoruz.
plt.figure(figsize=(8, 8))
plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha="center", va="center", color='white')
plt.grid('minor')
plt.title("Yapay Zekanın Kendi Kendine Öğrendiği Harf Haritası")
plt.show()


#modeli büyütme: daha büyük embedding ve daha fazla hidden layer
emb_dim = 10  #harfleri 2 sayıyla değil, 10 sayıyla ifade ediyoruz
hidden_size = 200  #nöron sayısını 100'den 200'e çıkarıyoruz

g = torch.Generator().manual_seed(2147483647)
C_large = torch.randn((vocab_size, emb_dim), generator=g)
W1_large = torch.randn((block_size * emb_dim, hidden_size), generator=g)  # 3 harf * 10 boyut = 30 girdi
b1_large = torch.randn(hidden_size, generator=g)
W2_large = torch.randn((hidden_size, vocab_size), generator=g)
b2_large = torch.randn(vocab_size, generator=g)

parameters_large = [C_large, W1_large, b1_large, W2_large, b2_large]
for p in parameters_large:
    p.requires_grad = True

# Büyük modelle asıl eğitim (daha zeki olması için adım sayısını 30.000 yapıyoruz)
for i in range(30000):
    ix = torch.randint(0, Xtr.shape[0], (32,))
    emb = C_large[Xtr[ix]]
    h = torch.tanh(emb.view(-1, block_size * emb_dim) @ W1_large + b1_large)
    logits = h @ W2_large + b2_large
    loss = F.cross_entropy(logits, Ytr[ix])

    for p in parameters_large: p.grad = None
    loss.backward()

    lr = 0.1 if i < 20000 else 0.01
    for p in parameters_large: p.data += -lr * p.grad

#gelişmiş modelin dev loss'u
emb_dev = C_large[Xdev]
h_dev = torch.tanh(emb_dev.view(-1, block_size * emb_dim) @ W1_large + b1_large)
logits_dev = h_dev @ W2_large + b2_large
dev_loss_large = F.cross_entropy(logits_dev, Ydev)
print(f"Büyütülmüş (10D Embedding, 200 Nöron) Dev Loss: {dev_loss_large.item():.4f}")

#isimler örnekleme
print("\n-Yeni modelin ürettiği isimler-")
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(10):
    out = []
    context = [0] * block_size  # '...' ile başlıyoruz
    while True:
        #seçili bağlamın embedding'ini çekip ağdan geçiriyoruz
        emb = C_large[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1_large + b1_large)
        logits = h @ W2_large + b2_large
        probs = F.softmax(logits, dim=1)

        #sonraki harfi seçiyor
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()

        #kayan pencereyi bir adım ileri kaydırıyoruz
        context = context[1:] + [ix]
        out.append(ix)

        if ix == 0:  #noktaya ulaştığında duruyor
            break

    print(''.join(itos[i] for i in out[:-1]))


#görev 5
print("""-----------------------
        Görev 5
-----------------------""")


print("-Başlangıçtaki loss'un yüksek olması-")

emb_dim = 10
hidden_size = 200
vocab_size = 27
block_size = 3

#rastgele, büyük başlangıç değerleri
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, emb_dim), generator=g)
W1 = torch.randn((block_size * emb_dim, hidden_size), generator=g)
b1 = torch.randn(hidden_size, generator=g)
W2 = torch.randn((hidden_size, vocab_size), generator=g)
b2 = torch.randn(vocab_size, generator=g)

#rastgele bir minibatch çekimi
ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
emb = C[Xtr[ix]]

#forward pass
hpreact = emb.view(-1, block_size * emb_dim) @ W1 + b1 #ham veriler
h = torch.tanh(hpreact)                                #tanh'tan çıkmış veriler
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr[ix])

print(f"Yüksek loss: {loss.item():.4f}")
print("olması gereken loss 3.29 civarıdır, ama şu an bundan çok yüksek.")
print('beklenen (uniform) loss:', -torch.log(torch.tensor(1/27)).item())
print('doymus (|h|>0.99) oran:', (h.abs() > 0.99).float().mean().item() * 100, '%')

#ölü nöronlar
plt.figure(figsize=(8, 5))
plt.hist(h.view(-1).tolist(), bins=50, color='#CBAACB')
plt.title("tanh doygunluğu (-1 ve 1'e yığılan ölü nöronlar)")
plt.show()


print("\n-Kaiming init ile tedavi etme-")

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, emb_dim), generator=g)

#kaiming formülü
fan_in = block_size * emb_dim #içeri giren kablo sayısı

# W1'i Kaiming katsayısı ile ölçekliyoruz: (5/3) / karekök(fan_in)
W1 = torch.randn((fan_in, hidden_size), generator=g) * (5/3) / (fan_in ** 0.5)
b1 = torch.randn(hidden_size, generator=g) * 0.01 #bias'ı ufaltıyoruz

# W2'yi küçültüyoruz, b2'yi de sıfır yapıyoruz ki başlangıçta çok yüksek olmasın
W2 = torch.randn((hidden_size, vocab_size), generator=g) * 0.01
b2 = torch.randn(vocab_size, generator=g) * 0

#forward pass
emb = C[Xtr[ix]]
hpreact = emb.view(-1, block_size * emb_dim) @ W1 + b1
h = torch.tanh(hpreact)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr[ix])

print(f"tedavi sonrası loss değeri: {loss.item():.4f}")
print('doymus (|h|>0.99) oran:', (h.abs() > 0.99).float().mean().item() * 100, '%')


#düzeltilmiş histogram
plt.figure(figsize=(8, 5))
plt.hist(h.view(-1).tolist(), bins=50, color='#CBAACB')
plt.title("Kaiming sonrası ortada toplanan nöronlar")
plt.show()


#görev 6
print("""-----------------------
        Görev 6
-----------------------""")

#batchnorm
vocab_size = 27
emb_dim = 10
block_size = 3
hidden_size = 100
batch_size = 32
max_steps = 10000

C  = torch.randn((vocab_size, emb_dim), generator=g)
W1 = torch.randn((block_size * emb_dim, hidden_size), generator=g) * (5/3) / ((block_size * emb_dim)**0.5) # Kaiming
W2 = torch.randn((hidden_size, vocab_size), generator=g) * 0.01
b2 = torch.randn(vocab_size, generator=g) * 0
#b1 artık yok


bngain = torch.ones((1, hidden_size))  #başlangıçta çarpım 1 (etkisiz)
bnbias = torch.zeros((1, hidden_size)) #başlangıçta toplama 0 (etkisiz)

#arka planda not alma
bnmean_running = torch.zeros((1, hidden_size))
bnstd_running = torch.ones((1, hidden_size))

parameters = [C, W1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True

#eğitim döngüsü
for i in range(max_steps):
    #torbadan rastgele 32'li minibatch çekme
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    #forward pass
    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)

    #hidden layer çarpımı (b1 yok)
    hpreact = embcat @ W1

    #batchnorm (eğitim sırasında)
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True)
    hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias

    #backward pass'e dahil olmaması için no_grad içinde. not defteri tutma kısmı (geçmişin %99.9'u şimdikinin %0.01..'i)
    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

    #aktivasyon ve tahmin
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)

    #backward pass ve güncelleme
    for p in parameters:
        p.grad = None
    loss.backward()

    #lr ayarı (ilk 5000 adım hızlı, sonra yavaş)
    lr = 0.1 if i < 5000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

print(f"Eğitim Sonu Minibatch Loss: {loss.item():.4f}")


#değerlendirme döngüsü (train,val)
@torch.no_grad()  #test yaparken modeli eğitmediğimiz için gradient hesaplanmaz
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
    }[split]

    emb = C[x]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1

    #tahmin sırasında batchnorm
    #burada .mean() veya .std() kullanmıyoruz.
    #onun yerine eğitimde doldurduğumuz running_mean ve running_std kullanıyoruz.
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias

    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(f"{split} Loss: {loss.item():.4f}")


split_loss('train')
split_loss('val')


#görev 7
print("""-----------------------
        Görev 7
-----------------------""")

def gorev_7():

    raw_lines = open('turkce_isimler.txt', 'r', encoding='utf-8').read().splitlines()
    words = []
    for line in raw_lines:
        w = line.strip()  #başındaki ve sonundaki boşlukları temizliyor
        w = w.replace('I', 'ı').replace('İ',
                                        'i').lower()  #asıl sorun büyük i ve ı harflerinin küçük harfe çevrilmesindeydi. onu manuel düzeltiyoruz.
        if w.isalpha() and len(
                w) > 0:  #veri temizleme: sadece harflerden oluşuyorsa ve uzunluğu 0'dan büyükse listeye ekliyoruz. birleşik isimleri eliyoruz.
            words.append(w)

    chars = sorted(list(set(''.join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    vocab_size = len(itos)

    block_size = 3  #önceki 3 harfe bakarak tahmin yapacak

    def build_dataset(words):
        X, Y = [], []
        for w in words:
            context = [0] * block_size
            for ch in w + '.':
                ix = stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]  #kayan pencereyi 1 adım ileri kaydırıyoruz
        return torch.tensor(X), torch.tensor(Y)

    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    Xtr, Ytr = build_dataset(words[:n1])
    Xdev, Ydev = build_dataset(words[n1:n2])
    Xte, Yte = build_dataset(words[n2:])

    emb_dim = 10
    hidden_size = 100
    batch_size = 32
    max_steps = 10000

    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((vocab_size, emb_dim), generator=g)

    # kaiming init ile w1
    W1 = torch.randn((block_size * emb_dim, hidden_size), generator=g) * (5 / 3) / ((block_size * emb_dim) ** 0.5)

    # w2yi küçültüp b2yi sıfırlıyoruz ki loss düşük başlasın
    W2 = torch.randn((hidden_size, vocab_size), generator=g) * 0.01
    b2 = torch.randn(vocab_size, generator=g) * 0

    # batchnorm parametreleri
    bngain = torch.ones((1, hidden_size))
    bnbias = torch.zeros((1, hidden_size))
    bnmean_running = torch.zeros((1, hidden_size))
    bnstd_running = torch.ones((1, hidden_size))

    parameters = [C, W1, W2, b2, bngain, bnbias]
    for p in parameters:
        p.requires_grad = True

    for i in range(max_steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]

        emb = C[Xb]
        embcat = emb.view(emb.shape[0], -1)
        hpreact = embcat @ W1

        bnmeani = hpreact.mean(0, keepdim=True)
        bnstdi = hpreact.std(0, keepdim=True)
        hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias

        with torch.no_grad():
            bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
            bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Yb)

        for p in parameters:
            p.grad = None
        loss.backward()

        lr = 0.1 if i < 5000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad

    @torch.no_grad()
    def split_loss(split):
        x, y = {'train': (Xtr, Ytr), 'val': (Xdev, Ydev)}[split]
        emb = C[x]
        embcat = emb.view(emb.shape[0], -1)
        hpreact = embcat @ W1

        # test yaparken running mean kullanıyoruz
        hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias

        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, y)
        print(f"{split} loss: {loss.item():.4f}")

    split_loss('train')
    split_loss('val')

    print("\nüretilen isimler:")
    g = torch.Generator().manual_seed(2147483647 + 10)

    for _ in range(10):
        out = []
        context = [0] * block_size
        while True:
            emb = C[torch.tensor([context])]
            embcat = emb.view(1, -1)
            hpreact = embcat @ W1
            hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
            h = torch.tanh(hpreact)
            logits = h @ W2 + b2

            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()

            context = context[1:] + [ix]
            out.append(ix)

            if ix == 0:
                break

        print(''.join(itos[i] for i in out[:-1]).capitalize())
gorev_7()