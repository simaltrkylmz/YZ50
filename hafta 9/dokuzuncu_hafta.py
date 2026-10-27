import regex as re
import tiktoken

#görev 1
#türkçe karakterler içeren cümle ve unicode karşılıkları
#unicode her karaktere bir numara veriyor. utf-8 de bu numarayı byte'lara çeviriyor.
#utf-8 tasarlanırken eski İngilizce karakter sistemiyle uyumlu olsun diye ilk 128 numara İngilizce harflere ve temel işaretlere ayrıldı.
#Türkçe harfler de bu ilk 128'in dışında kaldığı için 2 byte'a ihtiyaç duyuyorlar.
#bu tablo sürekli değişiyor, yeni karakterler ekleniyor. ayrıca 150 binden fazla karakter barındırıyor. direkt vocab olarak kullanmak bu yüzden kötü bir fikir.
sentence = "Bu kış soğuk geçecek diyorlar"

for char in sentence:
    code_point = ord(char) #Unicode'ta her karakterin numara karşılığı (kod noktası)
    byte_list = list(char.encode("utf-8"))
    print(f"{char!r:6} kod noktası: {code_point:>6} | UTF-8: {str(byte_list):<22} | {len(byte_list)} byte")

print(f"\n{len(sentence)} harf, {len(sentence.encode('utf-8'))} byte")

#görev 2
#Byte Pair Encoding (BPE)
#ardışık ikilileri sayma ve en sık geçenleri birleştirme.

def get_stats(ids):
    #yan yana gelen her ikilinin kaç kez geçtiğini sayıyor
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    #ids içindeki her pair'i idx ile değiştiriyor
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]: #len-1 kontrolü, son karakterdeki ikili aramayı atlıyor
            newids.append(idx)   #ikili bulundu, yeni token'ı ekle
            i += 2               #iki elemanı birden atla (çünkü ikincisi de bu oluşturduğumuz token'ın bir parçası)
        else:
            newids.append(ids[i])  #ikili değil, elemanı olduğu gibi ekle
            i += 1
    return newids

vocab_size = 512                 # istediğimiz vocab boyutu
num_merges = vocab_size - 256    # 256 byte zaten var, kalanı birleştirmelerle dolduracağız

text = open('turkce.txt', encoding='utf-8').read()
ids = list(text.encode('utf-8'))               # metni byte listesine çevir
merges = {}                                    # (token1, token2) -> yeni token numarası
vocab = {i: bytes([i]) for i in range(256)}    # token numarası -> byte'ları (yazdırmak için)

def show(b):
    #byte'ları okunabilir metne çeviriyor
    return b.decode('utf-8', errors='replace')

for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)   #en sık geçen ikili
    idx = 256 + i                      #yeni token'ın numarası
    ids = merge(ids, pair, idx)  #bulup değiştirme
    merges[pair] = idx #öğrenilen kurallar listesi (sonda elimizde kalan)
    vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
    if i < 20: #ilk 20'yi yazdırma
        print(f"{i+1:2d}. {show(vocab[pair[0]])!r} + {show(vocab[pair[1]])!r}"
              f" -> {show(vocab[idx])!r}  ({stats[pair]:,} kez)")

num_bytes = len(text.encode('utf-8'))
print(f"\n{num_bytes:,} byte -> {len(ids):,} token | sıkıştırma oranı: {num_bytes / len(ids):.2f}")

#görev 3
#encoding ve decoding
def decode(ids):
    #token'lar -> metin
    b = b"".join(vocab[idx] for idx in ids)        # her token'ın byte'larını uç uca ekle
    return b.decode("utf-8", errors="replace")     #byte'ları metne çevir, tek başına bişey ifade etmeyenleri replace kullanarak temsil et.

def encode(text):
    #metin -> token'lar
    #sırayla gitmek gerekiyor.
    ids = list(text.encode("utf-8"))      # önce metni byte'lara çevir
    for pair, idx in merges.items():      # defterdeki her değiştirmeyi, öğrenme sırasıyla
        ids = merge(ids, pair, idx)       # bul ve değiştir
    return ids

example = "Bu sene yurt dışına çıkmak istiyorum🙏 ."
coded = encode(example)

print("byte sayısı :", len(example.encode("utf-8")))
print("token sayısı:", len(coded))
print("token'lar   :", coded)
print("parçalar    :", [decode([t]) for t in coded])
print("geri çevrilmiş:", decode(coded))
print("aynı mı?    :", decode(coded) == example)

#görev 4
#GPT4 regex
GPT4_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
# İngilizce kısaltma ekleri, önündeki boşlukla birlikte kelime, en fazla 3 basamaklı sayılar, noktalama işaretleri, boşluklar ve satır sonları

#İngilizce kısaltma ekleri 's, 'd, 'm, 't, 'll, 've, 're gibi olduğu için Türkçede sıkıntı çıkıyor.
for kelime in ["Ankara'dan", "ANKARA'DAN","İstanbul'da", "Ali'nin", "Ahmet'te", "Ankara’dan"]:
    print(kelime, "->", re.findall(GPT4_PATTERN, kelime))

def get_stats(ids, counts=None):
    #yan yana gelen ikilileri sayıyor. counts verilirse onun üzerine ekliyor
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

parcalar = re.findall(GPT4_PATTERN, text)
ids = [list(p.encode("utf-8")) for p in parcalar]   # her parça ayrı bir byte listesi

merges = {}
vocab = {i: bytes([i]) for i in range(256)}

for i in range(num_merges):
    stats = {}
    for parca in ids:                 # 1. say: her parçanın ikililerini aynı sözlüğe ekle
        get_stats(parca, stats)
    pair = max(stats, key=stats.get)  # 2. en çok geçeni seç
    idx = 256 + i
    ids = [merge(parca, pair, idx) for parca in ids]   # 3. her parçada ayrı ayrı bul-değiştir
    merges[pair] = idx
    vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
    if i < 20:
        print(f"{i+1:2d}. {show(vocab[pair[0]])!r} + {show(vocab[pair[1]])!r}"
              f" -> {show(vocab[idx])!r}  ({stats[pair]:,} kez)")

num_tokens = sum(len(parca) for parca in ids)
print(f"\n{num_bytes:,} byte -> {num_tokens:,} token | sıkıştırma oranı: {num_bytes / num_tokens:.2f}")

def encode_regex(text):
    #metni önce gpt-4 deseniyle böl, sonra her parçayı ayrı encode et
    ids = []
    for part in re.findall(GPT4_PATTERN, text):
        part_ids = list(part.encode("utf-8"))
        for pair, idx in merges.items():
            part_ids = merge(part_ids, pair, idx)
        ids.extend(part_ids)
    return ids

#görev 5
#Türkçe ve İngilizce paragraf karşılaştırması
tr_text = ("Geçen hafta sonu arkadaşlarımla birlikte Kapadokya'ya gittik. "
           "Sabah erkenden kalkıp balonların gökyüzüne yükselişini izledik. "
           "Öğleden sonra yer altı şehirlerini gezdik ve yerel bir restoranda testi kebabı yedik. "
           "Akşam olduğunda hepimiz çok yorulmuştuk, ama bu yolculuğu uzun süre unutamayacağımızı biliyorduk.")

en_text = ("Last weekend, my friends and I went to Cappadocia. "
           "We got up early in the morning and watched the balloons rise into the sky. "
           "In the afternoon, we explored the underground cities and ate pottery kebab at a local restaurant. "
           "By evening, we were all very tired, but we knew we would not forget this trip for a long time.")

enc = tiktoken.get_encoding("cl100k_base")   # gpt-4'ün tokenizer'ı

for name, text in [("Türkçe", tr_text), ("İngilizce", en_text)]:
    tokens = enc.encode(text)
    words = len(text.split())
    print(f"{name:10} {words:3d} kelime | {len(tokens):3d} token | kelime başına {len(tokens) / words:.2f} token")

own_tokens = encode_regex(tr_text)
words = len(tr_text.split())
print(f"kendi tokenizer'ım: {words} kelime | {len(own_tokens)} token | kelime başına {len(own_tokens) / words:.2f} token")


for word in [" unutamayacağımızı", " arkadaşlarımla", " friends", " afternoon"]:
    print(f"{word!r}")
    print("   GPT-4  :", [enc.decode([t]) for t in enc.encode(word)])
    print("   benimki:", [decode([t]) for t in encode_regex(word)])


