import torch
import torch.nn as nn
from torch.nn import functional as F

#görev 1
#hazırlık
batch_size = 32       # aynı anda işlenen dizi sayısı
block_size = 8        # modelin geriye bakabileceği en fazla harf sayısı
n_embd = 32           # her harfi temsil eden vektörün boyutu
max_iters = 10000     # eğitim adımı (geçen haftayla aynı, adil karşılaştırma için)
eval_interval = 1000  # kaç adımda bir loss yazdırılacak
eval_iters = 200      # loss ortalaması için kaç batch kullanılacak
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#veri düzenleme
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


#modeller

class Head(nn.Module):
    #tek self-attention head

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                                              # (B, T, head_size)
        q = self.query(x)                                            # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5            # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # geleceği gizle
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)                                            # (B, T, head_size)
        return wei @ v                                               # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    #paralel çalışan birden çok head, çıktıları yan yana birleştiriliyor.

    def __init__(self, num_heads, head_size):
        super().__init__()
        # normal liste değil ModuleList: yoksa optimizer head'leri göremez
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        #birleştiriyoruz
        return torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, num_heads * head_size)

#görev 2 eklemesi: feedforward
class FeedForward(nn.Module):
#her harf topladığı bilgiyi tek başına işliyor

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential( #içine verdiğimiz katmanları sırayla çalıştırıyor: önce Linear, sonra ReLU

            nn.Linear(n_embd, n_embd),
            nn.ReLU(), #tanh'ın daha basit ve daha hızlı bir alternatifi
            #negatif sayılar 0 oluyor, pozitiflere dokunulmuyor. bir eşik koyuyor. eşiği geçmediysen sinyal verme diyor.
        )

    def forward(self, x):
        return self.net(x)

#görev 2 eklemesi: block sınıfı
class Block(nn.Module):
#transformer bloğu: önce iletişim (attention), sonra hesaplama (feedforward)

    def __init__(self, n_embd, n_head,residual=False):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.residual = residual

    def forward(self, x):
        if self.residual:
            x = x + self.sa(x)      # x'e attention'ın düzeltmesini ekle
            x = x + self.ffwd(x)    # x'e feedforward'ın düzeltmesini ekle
        else:
            x = self.sa(x) #residual'dan öncesi. direkt eski hali yok oluyor.
            x = self.ffwd(x)
        return x

class TransformerModel(nn.Module):
#blokları üst üste dizen model

    def __init__(self, n_head, n_layer,residual=False):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head, residual),
            Block(n_embd, n_head, residual),
            Block(n_embd, n_head, residual),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                                # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb                                                    # (B, T, n_embd)
        x = self.blocks(x)                                                       # (B, T, n_embd)
        logits = self.lm_head(x)                                                 # (B, T, vocab_size)
        return logits, hesapla_loss(logits, targets)



#loss hesaplama
def hesapla_loss(logits, targets):
    """(B, T, C) logits'i cross_entropy'nin istediği (B*T, C) şekline çevirip loss hesaplıyor"""
    if targets is None:
        return None
    B, T, C = logits.shape
    return F.cross_entropy(logits.view(B*T, C), targets.view(B*T))


class BigramLanguageModel(nn.Module):
    #bigram modeli

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)
        return logits, hesapla_loss(logits, targets)


class HeadAttention(nn.Module):
    #n_head=1 -> geçen haftaki tek head (32), n_head=4 -> 4 tane 8'lik head

    def __init__(self, n_head,ff=False):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd) if ff else None #bir açma kapama tuşu ekliyoruz. varsayılan False, yani istediğimizde çalışacak
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                                # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb                                                    # (B, T, n_embd)
        x = self.sa_heads(x) # (B, T, n_embd)
        if self.ffwd is not None:
            x = self.ffwd(x)
        logits = self.lm_head(x)                                                 # (B, T, vocab_size)
        return logits, hesapla_loss(logits, targets)


#eğitim ve metin üretme
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model, isim):
    #modeli eğitiyor, ilerlemeyi yazdırıyor, son val loss'u döndürüyor. 3 tane kullanacağımız için bir fonksiyon olarak yazdık.
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"\n=== {isim} | {n_param:,} parametre ===")


    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"Adım {iter:5d} | Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f}")
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


    losses = estimate_loss(model)
    print(f"Son val loss: {losses['val']:.4f}")
    return losses['val']


@torch.no_grad()
def generate(model, max_new_tokens=300):
    #yeni satır karakteriyle (0) başlayıp metin üretiyor
    model.eval()
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]  # pozisyon tablosu block_size satırlık, son 8 harfi ver
        logits, _ = model(idx_cond)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    model.train()
    return decode(idx[0].tolist())


#
sonuclar = {}
"""
# her modelden önce aynı seed: hepsi aynı koşullarda başlasın
torch.manual_seed(1337)
sonuclar['bigram'] = train(BigramLanguageModel(), 'bigram')

torch.manual_seed(1337)
sonuclar['tek head'] = train(HeadAttention(n_head=1), 'tek head (1 x 32)')

torch.manual_seed(1337)
model_multi = HeadAttention(n_head=4)
sonuclar['multi-head'] = train(model_multi, 'multi-head (4 x 8)')
print("\nmulti-head modelinin yazdığı metin:")
print(generate(model_multi))"""

"""torch.manual_seed(1337)
sonuclar['+ feedforward'] = train(HeadAttention(n_head=4, ff=True), '+ feedforward')

#residual olmadan bloklar
torch.manual_seed(1337)
sonuclar['3 blok'] = train(TransformerModel(n_head=4, n_layer=3), '3 blok (residual yok)')
"""
#residual eklendikten sonra
torch.manual_seed(1337)
sonuclar['+ residual'] = train(TransformerModel(n_head=4, n_layer=3, residual=True), '3 blok + residual')

#sonuçlar
print(f"{'model':<20} | {'val loss':>8}")
print("-" * 32)
for isim, val in sonuclar.items():
    print(f"{isim:<20} | {val:>8.4f}")