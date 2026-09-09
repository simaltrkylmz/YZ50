import torch


#görev 1

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