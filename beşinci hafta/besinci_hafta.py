#görev 1
print("""-----------------------
        Görev 1
-----------------------""")

import torch
import random
import torch.nn.functional as F

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

hprebn = embcat @ W1 + b1

#batchnorm adımları
bnmeani = hprebn.mean(0, keepdim=True) #32 örneğin ortalaması
bndiff = hprebn - bnmeani #her veriden ortalamayı çıkarıyoruz.
bndiff2 = bndiff**2
bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) #/n-1: bessel's correction
bnvar_inv = (bnvar + 1e-5)**-0.5 #çarpma bilgisayarda daha hızlı olduğu için genelde bölme için tersini alıp çarparız.
bnraw = bndiff * bnvar_inv #sıfıra merkezlenmiş veriyi standart sapmaya bölüyoruz.
hpreact = bngain * bnraw + bnbias #modele esneklik sağlamak için saf verimizi modelin kendi kendine eğitebildiği bngain (genişletme) ile çarpıp bnbias (kaydırma) ile topluyoruz.

h = torch.tanh(hpreact)
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
          norm_logits, logit_maxes, logits, h, hpreact,
          bnraw, bnvar_inv, bnvar, bndiff2, bndiff, bnmeani, hprebn, embcat, emb]:
    t.retain_grad()

loss.backward()
print(f"PyTorch Loss: {loss.item():.4f}")

#elle türev hesapları

#formülleri hatırlamak için üstlerine yazdım.

#loss = -logprobs[range(n), Yb].mean()
dlogprobs= torch.zeros_like(logprobs) #logprobs ile aynı boyutta sıfırlarla dolu matris
dlogprobs[range(n), Yb]=-1.0/n #logprobstaki sayılarla oynarsak loss'u nasıl etkilerler
cmp("logprobs", dlogprobs,logprobs) #kontrol fonksiyonu

#logprobs = probs.log()
dprobs= (1.0/probs) * dlogprobs #dloss/dprobs= (dloss/dlogprobs) * (dlogprobs/dprobs) : zincir kuralı
cmp("probs", dprobs,probs)

#probs = counts * counts_sum_inv
dcounts_sum_inv= (counts * dprobs).sum(1, keepdim=True)
#forward pass'te counts_sum_inv 30 harfe de etki etmesi için pytorch tarafından 32x30 boyutuna genişletiliyor. bu yüzden geri dönerken 30 farklı koldan gelen etkiyi tek sütunda topluyoruz.
cmp("counts_sum_inv", dcounts_sum_inv,counts_sum_inv)

#counts_sum_inv = counts_sum**-1
dcounts_sum= -counts_sum**-2 * dcounts_sum_inv
cmp("counts_sum", dcounts_sum,counts_sum)

#probs = counts * counts_sum_inv
#counts_sum = counts.sum(1, keepdims=True) #toplamın türevi 1
#counts iki ayrı yerde. bu yüzden geriye dönerken bu iki yoldan gelen türevleri toplamamız gerekiyor.
dcounts= counts_sum_inv * dprobs
dcounts+= torch.ones_like(counts) * dcounts_sum
cmp("counts", dcounts,counts)

#counts = norm_logits.exp()
dnorm_logits = dcounts * counts #e^x'in türevi yine kendisidir. bu da counts değişkeni içinde
cmp("norm_logits", dnorm_logits,norm_logits)

#norm_logits = logits - logit_maxes
dlogit_maxes= (-1 * dnorm_logits).sum(1,keepdim=True) #yine genişletme
cmp('logit_maxes', dlogit_maxes, logit_maxes)

#norm_logits = logits - logit_maxes
#logit_maxes = logits.max(1, keepdim=True).values
#sadece en yüksek değerli harfi içeri alıyoruz, kalanların etkisi yok.
dlogits= 1 * dnorm_logits
#sadece en yüksek olan değeri alıp üstten gelen türevle çarpıyoruz
dlogits+= F.one_hot(logits.max(1).indices, num_classes=logits.shape[1])*dlogit_maxes
cmp('logits', dlogits,logits)

#logits = h @ W2 + b2
#batchnorm kısmı
dh = dlogits @ W2.T
dW2 = h.T @ dlogits
db2=dlogits.sum(0) #bu sefer yan yana değil yukarıdan aşağıya toplayarak bir satıra indiriyoruz.
cmp('h', dh, h)
cmp('W2', dW2, W2)
cmp('b2', db2, b2)

#h = torch.tanh(hpreact)
#tanh'ın türevi 1- tanh(x)**2'dir. burada tanh(x)'i zaten h olarak yazdık
dhpreact= (1.0-h**2)*dh
cmp('hpreact', dhpreact,hpreact)

#hpreact = bngain * bnraw + bnbias
dbnbias=dhpreact.sum(0,keepdim=True)
dbngain=(bnraw*dhpreact).sum(0,keepdim=True)
dbnraw=bngain*dhpreact #bnraw zaten 32 satırlık tablo olduğu için broadcasting yok
cmp('bnbias', dbnbias,bnbias)
cmp('bngain', dbngain,bngain)
cmp('bnraw', dbnraw,bnraw)

#bnraw = bndiff * bnvar_inv
dbnvar_inv=(bndiff*dbnraw).sum(0,keepdim=True)
cmp('bnvar_inv', dbnvar_inv, bnvar_inv)

#bnvar_inv = (bnvar + 1e-5)**-0.5
dbnvar= (-0.5 *(bnvar+1e-5)**-1.5) * dbnvar_inv
cmp('bnvar', dbnvar,bnvar)

#bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) (1/n-1 burada sabitimiz: bessel's correction)
dbndiff2= (1.0/(n-1))*torch.ones_like(bndiff2)*dbnvar #toplamanın yerel türevi 1'dir.
cmp('bndiff2', dbndiff2,bndiff2)

#bndiff2 = bndiff**2
#bnraw = bndiff * bnvar_inv (bndiff iki yerde kullanılıyor.)
dbndiff=2*bndiff*dbndiff2
dbndiff+= bnvar_inv*dbnraw
cmp('bndiff', dbndiff,bndiff)

#bndiff = hprebn - bnmeani
#bnmeani = hprebn.mean(0, keepdim=True): bu aslında 1/n * hprebn.sum demek
dbnmeani= (-dbndiff).sum(0, keepdim=True) #forward pass'te bnmeani bir satırlık ortalamaydı ama 32 satırlık hprebn için 32 kere kopyalandı.
cmp('bnmeani', dbnmeani,bnmeani)

dhprebn= 1*dbndiff
dhprebn+=(1.0/n) * torch.ones_like(hprebn)*dbnmeani
cmp('hprebn', dhprebn,hprebn)

#hprebn = embcat @ W1 + b1
db1=dhprebn.sum(0)
dW1=embcat.T@dhprebn
dembcat=dhprebn@W1.T
cmp('b1',db1,b1)
cmp('W1',dW1,W1)
cmp('embcat', dembcat, embcat)

#emb = C[Xb]
#embcat = emb.view(emb.shape[0], -1)
#burada emb matrisinin boyutunu değiştirmiştik. şimdi onu eski haline döndürüyoruz.
demb=dembcat.view(emb.shape)
cmp('emb',demb,emb)

dC=torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix=Xb[k,j]
        dC[ix]+=demb[k,j]
cmp('C', dC,C)
