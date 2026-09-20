#görev 1
print("""-----------------------
        Görev 1
-----------------------""")

import torch
import torch.nn.functional as F
import random

#veri seti hazırlığı (geçen haftaların aynısı)
words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)

block_size = 3

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
Xtr, Ytr = build_dataset(words[:n1]) #sadece eğitim setiyle uğraşıyoruz



emb_dim = 10
hidden_size = 100
batch_size = 32
n = batch_size #formüllerdeki 'n' (32)

g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, emb_dim), generator=g)

#b1'i geri getiriyoruz.
W1 = torch.randn((block_size * emb_dim, hidden_size), generator=g) * (5/3) / ((block_size * emb_dim)**0.5)
b1 = torch.randn(hidden_size, generator=g) * 0.01
W2 = torch.randn((hidden_size, vocab_size), generator=g) * 0.01
b2 = torch.randn(vocab_size, generator=g) * 0

#bngain ve bnbias'a ufak rastgelelik ekliyoruz.
bngain = torch.randn((1, hidden_size), generator=g) * 0.1 + 1.0
bnbias = torch.randn((1, hidden_size), generator=g) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True


ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]

#kıyaslama fonksiyonumuz
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item() #bizim bulduğumuz türevle pytorch'un bulduğu tıpatıp aynı mı?
    app = torch.allclose(dt, t.grad) #sayılar birbirine çok yakınsa true der. yuvarlama farkı olduğunu anlarız.
    maxdiff = (dt - t.grad).abs().max().item() #bizim bulduğumuzla pytorch'un bulduğu arasındaki en büyük fark ne?
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')


#forward pass
emb = C[Xb]
embcat = emb.view(emb.shape[0], -1)

hpreact = embcat @ W1 + b1

#batchnorm adımları
bnmeani = hpreact.mean(0, keepdim=True) #32 örneğin ortalaması
bndiff = hpreact - bnmeani #her veriden ortalamayı çıkarıyoruz.
bndiff2 = bndiff**2
bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) #/n-1: bessel's correction
bnvar_inv = (bnvar + 1e-5)**-0.5 #çarpma bilgisayarda daha hızlı olduğu için genelde bölme için tersini alıp çarparız.
bnraw = bndiff * bnvar_inv #sıfıra merkezlenmiş veriyi standart sapmaya bölüyoruz.
hpreact_fast = bngain * bnraw + bnbias #modele esneklik sağlamak için saf verimizi modelin kendi kendine eğitebildiği bngain (genişletme) ile çarpıp bnbias (kaydırma) ile topluyoruz.

h = torch.tanh(hpreact_fast)
logits = h @ W2 + b2

#cross entropy adımları
logit_maxes = logits.max(1, keepdim=True).values #her satırın kendi içindeki maksimumu
norm_logits = logits - logit_maxes #her örneğin her harfinden o örneğin kendi maksimumunu çıkarıyoruz.
counts = norm_logits.exp() #e üzeri yazıyoruz.
counts_sum = counts.sum(1, keepdims=True) #yine dim 1, her örneğin kendi harflerinin toplamı
counts_sum_inv = counts_sum**-1 #toplamın -1. kuvveti
probs = counts * counts_sum_inv #olasılık hesaplaması
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()

#pytorch ile otomatik yapma
for p in parameters:
    p.grad = None

#bütün ara adımları hafızada tutması için uyarıyoruz
for t in [logprobs, probs, counts_sum_inv, counts_sum, counts,
          norm_logits, logit_maxes, logits, h, hpreact_fast,
          bnraw, bnvar_inv, bnvar, bndiff2, bndiff, bnmeani, hpreact, embcat, emb]:
    t.retain_grad()

loss.backward()
print(f"PyTorch Loss: {loss.item():.4f}")

