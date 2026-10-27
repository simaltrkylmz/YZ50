import torch
import torch.nn.functional as F
import random

#Türkçe veri düzenlemesi
words = []
for line in open('turkce_isimler.txt', 'r', encoding='utf-8').read().splitlines():
    w = line.strip().replace('I', 'ı').replace('İ', 'i').lower()
    if w.isalpha() and len(w) > 0:
        words.append(w)

chars = sorted(set(''.join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)
print('isim sayisi:', len(words), ' alfabe boyutu:', vocab_size)

def turkce_buyuk_ilk_harf(s):
    if s and s[0] == 'i':
        return 'İ' + s[1:]
    return s.capitalize()

def build_dataset(words_list, block_size):
    X, Y = [], []
    for w in words_list:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

random.seed(42)
words_shuffled = words.copy()
random.shuffle(words_shuffled)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

# iki farklı bağlam uzunluğu için
Xtr8, Ytr8 = build_dataset(words_shuffled[:n1], 8)
Xdev8, Ydev8 = build_dataset(words_shuffled[n1:n2], 8)
Xtr3, Ytr3 = build_dataset(words_shuffled[:n1], 3)
Xdev3, Ydev3 = build_dataset(words_shuffled[n1:n2], 3)
print(f"train: {Xtr8.shape[0]}, dev: {Xdev8.shape[0]}")


#oluşturduğumuz sınıflar
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    def __call__(self, x):
        if self.training:
            dim = 0 if x.ndim == 2 else (0, 1)
            xmean = x.mean(dim, keepdim=True)
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


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []


class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
    def __call__(self, ix):
        self.out = self.weight[ix]
        return self.out
    def parameters(self):
        return [self.weight]


class FlattenConsecutive:
    def __init__(self, n):
        self.n = n
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out
    def parameters(self):
        return []


class Sequential:
    def __init__(self, layers):
        self.layers = layers
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


#yardımcı fonksiyonlar
#loss hesaplama
@torch.no_grad()
def eval_loss(model, X, Y):
    for layer in model.layers:
        layer.training = False
    loss = F.cross_entropy(model(X), Y).item()
    for layer in model.layers:
        layer.training = True
    return loss

#modeli eğitme
def train_model(model, Xtr, Ytr, Xdev, Ydev, steps, batch_size=32, print_every=2000):
    parameters = model.parameters()
    for p in parameters:
        p.requires_grad = True
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
        loss = F.cross_entropy(model(Xtr[ix]), Ytr[ix])
        for p in parameters:
            p.grad = None
        loss.backward()
        lr = 0.1 if i < steps * 0.75 else 0.01
        for p in parameters:
            p.data -= lr * p.grad
        if i % print_every == 0:
            print(f'{i:6d}/{steps}  train {eval_loss(model, Xtr, Ytr):.4f}  dev {eval_loss(model, Xdev, Ydev):.4f}')
    return sum(p.nelement() for p in parameters)

@torch.no_grad()
#isim üretme
def generate(model, block_size, n=10, seed=123):
    g = torch.Generator().manual_seed(seed)
    for layer in model.layers:
        layer.training = False
    for _ in range(n):
        out = []
        context = [0] * block_size
        while True:
            probs = F.softmax(model(torch.tensor([context])), dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            if ix == 0:
                break
            out.append(itos[ix])
        print(turkce_buyuk_ilk_harf(''.join(out)))
    for layer in model.layers:
        layer.training = True


#hafta 4 MLP'si: adil karşılaştırma için
torch.manual_seed(42)
mlp3 = Sequential([
    Embedding(vocab_size, 10),
    FlattenConsecutive(3),
    Linear(10 * 3, 200, bias=False), BatchNorm1d(200), Tanh(),
    Linear(200, vocab_size),
])
print("Hafta 4 - 3 bağlam")
p_mlp3 = train_model(mlp3, Xtr3, Ytr3, Xdev3, Ydev3, steps=10000)


#Wavenet
torch.manual_seed(42)
n_embd, n_hidden = 24, 128
wave8 = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
])
print("Wavenet - 8 bağlam")
p_wave8 = train_model(wave8, Xtr8, Ytr8, Xdev8, Ydev8, steps=20000)

#sonuç tablosu
print(f"\n{'Model':<20}{'Parametre':<12}{'Train':<10}{'Dev':<10}")
print(f"{'Baglam 3 MLP':<20}{p_mlp3:<12}{eval_loss(mlp3, Xtr3, Ytr3):<10.4f}{eval_loss(mlp3, Xdev3, Ydev3):<10.4f}")
print(f"{'Baglam 8 WaveNet':<20}{p_wave8:<12}{eval_loss(wave8, Xtr8, Ytr8):<10.4f}{eval_loss(wave8, Xdev8, Ydev8):<10.4f}")

print('\nBaglam 3 MLP isimleri:')
generate(mlp3, 3)
print('\nWaveNet isimleri:')
generate(wave8, 8)
