import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

#önceki haftaki koddan gerekli kısımlar
words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos) # 27

#X ve Y veri setini kurma
block_size = 8 #bağlam penceresi
def build_dataset(words_list):
    X, Y = [], []
    for w in words_list:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

#veriyi bölme
random.seed(42)
words_shuffled = words.copy()
random.shuffle(words_shuffled) #sözlükteki isimler sıralı olduğu için onları karıştırıyoruz.
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words_shuffled[:n1])
Xdev, Ydev = build_dataset(words_shuffled[n1:n2])
Xte, Yte = build_dataset(words_shuffled[n2:])
print(f"Veri setleri -> train: {Xtr.shape[0]}, dev: {Xdev.shape[0]}, test: {Xte.shape[0]}\n")


#görev 1
#daha öncesinde elle yazdığımız işlemler büyük modellerde mantıklı değil bu yüzden her katmanı bir sınıf haline getiriyoruz.
class Linear:
    #x @ W + b işlemini yapıyor.
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5 #biraz sadeleştirilmiş kaiming init
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self): #bu classın öğrenilecek parametreleri
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # Öğrenilebilir parametreler
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        # Koşan istatistikler (Inference için)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            #görev 4 eklemesi: 3 boyut için mean hesabı düzeltmesi
            if x.ndim==2:
                dim=0
            elif x.ndim==3:
                dim=(0,1)
            xmean = x.mean(dim, keepdim=True) # Şimdilik sadece 0. boyutta (batch)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.squeeze()
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


#tanh fonksiyonuna sıkıştırıyor.
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return [] #sadece kendine gelen sayıları -1 ile 1 arasına sıkıştırıyor.


#harfleri bilgisayarın anladığı vektörlere dönüştürüyor.
class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, ix):
        self.out = self.weight[ix]
        return self.out

    def parameters(self):
        return [self.weight]


#bütün harfleri tek vektörde düzleştiriyoruz

#görev 3 flatten class'ı
#ikişerli birleştirme yapıyoruz
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)
        if x.shape[1] == 1: #eğer gruplama sonucu tek grup kaldıysa (T/n) 1 ise B,1,C*n yerine B,C*n yapıyor.
            x = x.squeeze(1)
        self.out = x
        return self.out
    def parameters(self):
        return []

""" görev 2 flatten class'ı
class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []
"""

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) #öncekinin çıktısı sonrakinin girdisi oluyor
        self.out = x
        return self.out

    def parameters(self):
        #bütün katmanların parametrelerini alıp bir listeye topluyoruz
        return [p for layer in self.layers for p in layer.parameters()]


#eğitim
n_embd = 24 #görev 5 güncellemesi
n_hidden= 128 #wavenet'i yazarken total parametrenin değişmemesi için değiştiriyoruz.
#n_hidden = 200 #wavenet'ten önceki hidden layer sayısı

torch.manual_seed(42)
#model tek bir Sequential kutusu oldu. eğitim döngüsü içeride ne olduğunu bilmiyor.
#görev 3 değişikliği
model = Sequential([
    Embedding(vocab_size, n_embd),
    # 8 harften başlıyoruz. Her FlattenConsecutive(2) T'yi yarıya indiriyor. 8-4-2-1
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size)
])


"""
model = Sequential([
    Embedding(vocab_size, n_embd),
    Flatten(),
    Linear(n_embd * block_size, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, vocab_size)
])
"""


#tüm parametreleri bir araya toplayıp gradient hesabına açıyoruz.
parameters = model.parameters()
print(sum(p.nelement() for p in parameters)) # Toplam parametre sayısını yazdırır
for p in parameters:
    p.requires_grad = True

#eğitim döngüsü
max_steps = 200000
batch_size = 32
lossi = []


for i in range(max_steps):
    #minibatch oluşturma
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    #forward pass
    #modelimizin içi gizli, sadece x'i verip logits'i alıyoruz
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    #lr güncelleme
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data -= lr * p.grad

    #her 10000 adımda bir yazdırıyoruz
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.item()) #loss'ları ekliyoruz
print(loss.item())

#loss eğrisi çizimini düzeltme
plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
plt.title("Eğitim Kaybı (Training Loss)")
plt.show()


#modelleri karşılaştırmak için her modelin aynı koşulda ölçülmüş loss'u olmalı.
#bu yüzden bütün veri setine bakıyoruz, rastgele tek bir minibatch'a değil.
@torch.no_grad()
def split_loss(split):
    x, y = {'train': (Xtr, Ytr), 'dev': (Xdev, Ydev), 'test': (Xte, Yte)}[split]
    for layer in model.layers:
        layer.training = False #batchnormda running mean ve running var ile hesaplaması gerektiğini söylüyoruz.
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())
    for layer in model.layers:
        layer.training = True

split_loss('train')
split_loss('dev')


"""
görev 1 çıktısı:
parametre sayısı: 12097
train 2.0621190071105957
dev 2.107567071914673
"""

#görev 2: bağlamı 3 harften 8 harfe çıkarıyoruz
"""
görev 2 çıktısı:
parametre sayısı: 22097
parametre sayısı 10000 arttı.
Flatten katmanının çıktısı 3*10'dan 8*10'a yükseldi. ilk Linear katmanının ağırlık matrisi de 30x200 boyutundan 80x200'e genişledi.
Bu 50 ek girişin her biri 200 nörona bağlandığı için 10000 yeni parametre eklenmiş oldu.
loss: 1.7623770236968994 (bir minibatch'in)
train 1.9282917976379395
dev 2.030041217803955
"""

Xb = Xtr[:32]
print("--- Wavenet Şekilleri ---")
x = Xb
for layer in model.layers:
    x = layer(x) # Veriyi sıradaki katmandan geçiriyoruz
    print(f"{layer.__class__.__name__:18s} : {tuple(x.shape)}")


"""
görev 4 çıktısı:
parametre sayısı: 22397

(düzeltmeden önce)
train 1.9531000852584839
dev 2.0360381603240967
running mean'in boyutu: torch.Size([4, 68])

(düzelttikten sonra)
train 1.9236152172088623
dev 2.025803565979004
running mean'in boyutu: torch.Size([68])
"""

""" görev 5 çıktısı:
parametre sayısı: 76579
train 1.7869821786880493
dev 1.9925611019134521
"""