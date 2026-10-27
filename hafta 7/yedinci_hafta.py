import torch
import torch.nn as nn
from torch.nn import functional as F

#görev 1
#veri setini hazırlama ve bigram modelini kurma
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
    y= torch.stack([data[i+1:i+block_size+1] for i in ix]) #x'in bir sağından başlıyoruz. amaç bir sonraki harfi tahmin etmek.
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
        #satırlar şu anki harfi, sütunlar "bundan sonra hangi harf gelmeli?" skorlarını (logits) tutuyor.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
    #eğitim verimizi alıp embedding tablosundan geçiriyor.
        logits = self.token_embedding_table(idx) #çıktı: (B, T, C)

        if targets is None:
            loss = None
        else:
            #PyTorch'un cross_entropy fonksiyonu 3 boyutlu matrisleri sevmez.
            #O yüzden B ve T boyutlarını alt alta düz bir listeye diziyoruz.
            B, T, C = logits.shape
            logits = logits.view(B*T, C) #örnek sayısı, sınıf sayısı şekline dönüştürülüyor
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
print("\n-Eğitimden önce üretilen metin-")
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
#öğrenme hızını 0.001 yapıyoruz. transformer'lar için genelde bu değer kullanılır.
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch_size = 32 #gerçek eğitime geçtiğimiz için paket boyutunu büyütüyoruz

print("\n-Bigram Eğitimi- \n")
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
tril = torch.tril(torch.ones(T, T)) #alt üçgen matrisi
wei = wei.masked_fill(tril == 0, float('-inf')) #geleceği sansürlüyoruz, -inf yapıyoruz.
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

#Görev 3
#Self Attention
torch.manual_seed(1337)
B, T, C = 4, 8, 32
n_embd = 32  # her token'ı temsil eden vektörün boyutu
head_size = 16

# gerçek veriden bir batch alıp token + pozisyon embedding'ini topluyoruz
xb, yb = get_batch('train')
xb = xb[:4]  # batch_size şu an 32, örnek için 4 tane yeter
B, T = xb.shape

tok_emb_table = nn.Embedding(vocab_size, n_embd) #harfleri temsil eden tablo
#attention da bir tür ağırlıklı toplama olduğu için kendi başına sırayı bilmez, bu yüzden konumları temsil eden tablo oluşturuyoruz.
pos_emb_table = nn.Embedding(block_size, n_embd)  # her pozisyon (0..7) için ayrı bir vektör
tok_emb = tok_emb_table(xb)  # (B,T,C) -> harf "ne"
pos_emb = pos_emb_table(torch.arange(T))  # (T,C)   -> harf "nerede"
x = tok_emb + pos_emb  # (B,T,C), broadcasting ile her batch'e ekleniyor
# hii kelimesinde son iki i'nin toplamları böylece farklı olmuş oluyor. konum vektörü olmasa aynı olacaklardı ama aynı değiller.

key = nn.Linear(n_embd, head_size, bias=False)  # "bende ne var?"
query = nn.Linear(n_embd, head_size, bias=False)  # "ben ne arıyorum?"
value = nn.Linear(n_embd, head_size, bias=False)  # "bana bakan olursa ona ne vereceğim?"

#her harf önce kendi q'sunu geçmişteki harflerin k'sıyla karşılaştırıp kime ne kadar bakacağına karar veriyor
#sonra bu ağırlıklara göre o harflerin v'lerinden bilgi topluyor.
k = key(x)  # (B,T,16)
q = query(x)  # (B,T,16)
#satır sütun çarpımı, q'nun satırları harfler, k'nın da satırları harfler, onların sütun olması için o boyutta transpose alıyoruz.
wei = q @ k.transpose(-2, -1) * head_size ** -0.5  # (B,T,16) @ (B,16,T) -> (B,T,T)
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))  # üst üçgen (gelecek) gizleniyor
wei = F.softmax(wei, dim=-1)
v = value(x)
out = wei @ v  # (B,T,T) @ (B,T,16) -> (B,T,16)
print("out şekli:", out.shape)


print("\nilk örneğin wei matrisi:")
print(wei[0])

# bir satırı okuyoruz: son token geçmişteki hangi harflere ne kadar bakıyor
ornek = xb[0].tolist()
print("\ndizi:", decode(ornek))
print(f"son harf {itos[ornek[-1]]} şu harflere bakıyor:")
for i, w in enumerate(wei[0, -1].tolist()):
    print(f"  {i}. {itos[ornek[i]]}: {w:.4f}")
#ağırlıklar henüz eğitilmedi, bu yüzden dağılım rastgele oluyor.

#neden head_size'ın kareköküne bölüyoruz?
torch.manual_seed(1337)
k = torch.randn(B, T, head_size)
q = torch.randn(B, T, head_size)
wei_bolmeden = q @ k.transpose(-2, -1)
wei_bolerek = wei_bolmeden * head_size ** -0.5

print(f"\nk varyansı: {k.var().item():.3f} | q varyansı: {q.var().item():.3f}")
print(f"bölmeden wei varyansı: {wei_bolmeden.var().item():.3f}")  # head_size (16) civarı
print(f"bölerek wei varyansı:  {wei_bolerek.var().item():.3f}")  # 1 civarı

# aynı satırın softmax çıktısını yan yana koyuyoruz
p_bolmeden = F.softmax(wei_bolmeden[0, -1], dim=-1)
p_bolerek = F.softmax(wei_bolerek[0, -1], dim=-1)
print("\npozisyon | bölmeden | bölerek")
for i in range(T):
    print(f"   {i}     |  {p_bolmeden[i].item():.3f}   |  {p_bolerek[i].item():.3f}")
print(f"en büyük ağırlık -> bölmeden: {p_bolmeden.max().item():.3f}, bölerek: {p_bolerek.max().item():.3f}")

# videodaki küçük örnek: skorlar büyüdükçe softmax tek bir değere yapışıyor (one-hot'a dönüyor)
s = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])
print("\nsoftmax(s):  ", F.softmax(s, dim=-1))
print("softmax(s*8):", F.softmax(s * 8, dim=-1))


#görev 4
#head'i bigram modelinin içine koyup eğitme
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        #tril matrisi öğrenilen bir şey değil. register_buffer: "bunu modelle birlikte sakla ama optimizer güncellemesin." diyor.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        #k.shape[-1] de head boyutu
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        #metin üretmenin başında dizi 1,2,.. harf uzunluğunda olabilir. [:T, :T], tril'in sol üst köşesinden TxT'lik kısmı alıyor.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

class SingleHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)       # 65 × 32
        self.position_embedding_table = nn.Embedding(block_size, n_embd)    # 8 × 32
        self.sa_head = Head(n_embd)    # head_size = 32
        #head'ten çıkan out da 32 sayı. ama loss'u hesaplamak için her harf için 65 skora ihtiyacımız var.
        #bu yüzden en sona 32 sayıyı 65 sayıya çeviren bir linear katmanı koyuyoruz.
        self.lm_head = nn.Linear(n_embd, vocab_size)                        # 32 → 65

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                  # (B, T, 32)
        pos_emb = self.position_embedding_table(torch.arange(T))   # (T, 32)
        x = tok_emb + pos_emb                                      # (B, T, 32)
        x = self.sa_head(x)                                        # (B, T, 32)
        logits = self.lm_head(x)                                   # (B, T, 65)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # metin üretme fonksiyonu
        for _ in range(max_new_tokens):
            #metin üretilirken uzunluk 8'i geçince pozisyon tablosunda sadece block_size kadar satır olduğu için sorun çıkıyor.
            #biz de zaten hep en sonki 8 harfle çalıştığımız için -block_size diyerek her zaman son 8 harfi modele veriyoruz.
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # sadece en son tahmin edilen harfin olasılıklarını alıyor.
            probs = F.softmax(logits, dim=-1)  # softmax ile yüzdelik ihtimallere çeviriyor.
            idx_next = torch.multinomial(probs, num_samples=1)  # olasılıklara göre rastgele 1 harf seçiyor.
            idx = torch.cat((idx, idx_next), dim=1)  # yeni harfi eski diziye ekliyor.
        return idx


#eğitim kısmı
torch.manual_seed(1337)
model = SingleHeadModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(10000):
    if iter % 1000 == 0:
        losses = estimate_loss()
        print(f"Adım {iter:4d} | Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
losses = estimate_loss()
head_val_loss = losses['val']
print(f"Bigram val loss:     {bigram_val_loss:.4f}")
print(f"Tek head val loss:   {head_val_loss:.4f}")
print(f"Düşüş:               {bigram_val_loss - head_val_loss:.4f}")

context = torch.zeros((1, 1), dtype=torch.long) #üretime "0" (yeni satır karakteri) ile başlıyoruz
print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))






