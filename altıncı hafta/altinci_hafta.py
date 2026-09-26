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
vocab_size = len(itos)  # 27

#X ve Y veri setini kurma
block_size = 3 #bağlam penceresi: önceki 3 harf
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
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

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


#görev 1
#daha öncesinde elle yazdığımız işlemler büyük modellerde mantıklı değil bu yüzden her katmanı bir sınıf haline getiriyoruz.
class Linear:
#x @ W + b işlemini yapıyor.
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5  #biraz sadeleştirilmiş kaiming init
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
            xmean = x.mean(0, keepdim=True)  # Şimdilik sadece 0. boyutta (batch)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
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
class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []


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
vocab_size = 27
n_embd = 10
n_hidden = 100
block_size = 3

#model tek bir Sequential kutusu oldu. eğitim döngüsü içeride ne olduğunu bilmiyor.
model = Sequential([
    Embedding(vocab_size, n_embd),
    Flatten(),
    Linear(n_embd * block_size, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, vocab_size)
])

#tüm parametreleri bir araya toplayıp gradient hesabına açıyoruz.
parameters = model.parameters()
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

#loss eğrisi çizimini düzeltme
plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
plt.title("Eğitim Kaybı (Training Loss)")
plt.show()


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



