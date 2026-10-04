import torch
import torch.nn as nn
from torch.nn import functional as F

#görev 1
text = open('input.txt', 'r',encoding='utf-8').read()
print(len(text))
chars=sorted(list(set(text)))
vocab_size=len(chars)
print(''.join(chars))
print(vocab_size)

stoi={ch:i for i,ch in enumerate(chars) }
itos={i:ch for i,ch in enumerate(chars) }
encode= lambda s: [stoi[c] for c in s] #string'i integerlara çeviriyor.
decode= lambda l: ''.join([itos[i] for i in l]) #integerları string'e çeviriyor.

print(encode("hii there"))
print(decode([46, 47, 47, 1, 58, 46, 43, 56, 43]))

data=torch.tensor(encode(text),dtype=torch.long)
print(data.shape, data.dtype)
print(data[:10])

#veri setini %90 ve %10 olarak ayırma
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]
print(f"train data: {len(train_data)}, val data: {len(val_data)}")

torch.manual_seed(1337)
batch_size=4
block_size=8

#block size kadar elementi daha iyi oturması için yazdırıyoruz.
"""x=train_data[:block_size]
y=train_data[1:block_size+1]
for t in range(block_size):
    context= x[:t+1]
    target=y[t]
    print(f"when input is {context}, the target is {target}")
output: ... when input is tensor([18, 47, 56, 57, 58,  1, 15]), the target is 47
            when input is tensor([18, 47, 56, 57, 58,  1, 15, 47]), the target is 58"""

#get_batch fonksiyonu
def get_batch(split):
    data=train_data if split=="train" else val_data #train veya val setinden rastgele batch üretimi
    ix=torch.randint(len(data)-block_size,(batch_size,)) #rastgele başlangıç indeksleri
    #uzunluk - block size yazıyoruz çünkü mesela sondan 3. harften başlarsa hata verir (block size kadar ileri gidemeyiz.)
    #(batch_size,) yazmamızın sebebi python'a 4 elemanlı 1 boyutlu rastgele sayılar dizisi istediğimizi anlatmak.

    x=torch.stack([data[i:i+block_size] for i in ix]) #seçtiğimiz noktalardan ileri doğru 8 harfi alıyoruz, yeni listede topluyoruz.
    y= torch.stack([data[i+1:i+block_size+1] for i in ix]) #x'in bir sağından başlıyor. amaç bir sonraki harfi tahmin etmek.
    return x,y

xb, yb = get_batch('train')
print("Girdilerin Şekli:", xb.shape)
#print(xb)
print("Hedeflerin Şekli:", yb.shape)
#print(yb)

#Bigram modeli
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #satırlar şu anki harfi, sütunlar bundan sonra hangi harf gelmeli? skorlarını (logits) tutuyor.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
    #eğitim verimizi alıp embedding tablosundan geçiriyor.
        logits = self.token_embedding_table(idx) #çıktı: (B, T, C)

        if targets is None:
            loss = None
        else:
            #PyTorch'un cross_entropy fonksiyonu 3 boyutlu matrisleri sevmez.
            #O yüzden B ve T boyutlarını (32,8) alt alta düz bir listeye diziyoruz. boyut (256,65) oluyor.
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        #metin üretme fonksiyonu
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] #sadece en son tahmin edilen harfin olasılıklarını alıyor.
            probs = F.softmax(logits, dim=-1) #softmax ile yüzdelik ihtimallere çeviriyor.
            idx_next = torch.multinomial(probs, num_samples=1) #olasılıklara göre rastgele 1 harf seçiyor.
            idx = torch.cat((idx, idx_next), dim=1) #yeni harfi eski diziye ekliyor.
        return idx

#modeli yaratıyoruz
model = BigramLanguageModel(vocab_size)
print("-Eğitimden önce üretilen metin-")
context = torch.zeros((1, 1), dtype=torch.long) #üretime "0" (yeni satır karakteri) ile başlıyoruz
print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))

#loss hesaplama fonksiyonu
#eğitim sırasında loss değerleri çok dalgalandığı için net bir ortalama görmek adına bu fonksiyonu kullanıyoruz.
@torch.no_grad() #türev hesabını kapatıyoruz ki hafızayı yormayalım.
def estimate_loss():
    out = {}
    model.eval() #modeli test moduna alıyor
    for split in ['train', 'val']:
        losses = torch.zeros(200)
        for k in range(200):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train() #modeli tekrar eğitim moduna alıyor
    return out

#eğitim döngüsü
#öğrenme hızını 0.0001 yapıyoruz. transformer'lar için genelde bu değer kullanılır.
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch_size = 32 #gerçek eğitime geçtiğimiz için paket boyutunu büyütüyoruz

print("Bigram Eğitimi \n")
for iter in range(10000):
    # Her 1000 adımda bir gidişatı ekrana yazdırıyoruz
    if iter % 1000 == 0:
        losses = estimate_loss()
        print(f"Adım {iter:4d} | Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f}")

    #eğitim kısmı
    xb, yb = get_batch('train') #veriyi getiriyor
    logits, loss = model(xb, yb) #tahmin edip hatayı buluyor
    optimizer.zero_grad(set_to_none=True) #eski türevleri temizliyor
    loss.backward() #hatanın kaynağını buluyor
    optimizer.step() #ağırlıkları (tabloyu) düzeltiyor

losses = estimate_loss()
print(f"\nSon Val Loss: {losses['val']:.4f}\n")
bigram_val_loss = losses['val']

#modelin eğitimden sonra yazdığı metni yazdırıyoruz
print("Bigram modelinin yazdığı metin")
context = torch.zeros((1, 1), dtype=torch.long) #üretime "0" (yeni satır karakteri) ile başlıyoruz
print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))


#görev 2
#bigram geçmişi bilmiyor. Biz de geçmişin ortalamasını 3 yolla hesaplıyoruz.
torch.manual_seed(1337)
B, T, C = 4, 8, 2          # batch, zaman (token sayısı), kanal (özellik boyutu)
x = torch.randn(B, T, C)

#1. yöntem: for döngüsü
xbow1 = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]          # 0'dan t'ye kadar olan tüm geçmiş
        xbow1[b, t] = torch.mean(xprev, 0)

#2. yöntem: tril- alt üçgen matris
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x              # (T,T) @ (B,T,C) -> (B,T,C), broadcasting ile

#3. yöntem: softmax
wei = torch.zeros((T, T)) #önce herkesin birbiriyle olan ilişkisi 0 diyoruz
tril = torch.tril(torch.ones(T, T)) #alt üçgen
wei = wei.masked_fill(tril == 0, float('-inf')) #geleceği sansürlüyoruz. matrisin alt üçgenini (geleceği) -inf yapıyoruz.
#çünkü mantıken 2. harf 3. harfi göremez, çünkü daha yazılmadı. 0'ları -inf yapıyoruz.
wei = F.softmax(wei, dim=-1) #sayıları önce e'nin üssü olarak yazıp toplama bölüyor.
#e^-inf=0 yaparak onlara %0 ihtimal verir. e^0=1 -> nötr olan bölgeleri 1 değerine dönüştürür ve satırdaki diğer 1'lerle eşit paylaştırır.
xbow3 = wei @ x #matris çarpımı (2.yöntem gibi)

#allclose ile karşılaştırma
"""print(torch.allclose(xbow1, xbow2))
print(torch.allclose(xbow1, xbow3))
böyle olunca false dedi. farkı yazdırınca da
 tensor(3.2363e-08)
tensor(3.2363e-08) bu çıktıyı aldım. çok çok küçük fark. o yüzden tolerans değerini biraz arttırıp tekrar yazdım."""

print(torch.allclose(xbow1, xbow2, atol=1e-6))
print(torch.allclose(xbow1, xbow3, atol=1e-6))

