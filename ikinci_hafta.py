import math
#görev 1'i ve 2'yi graphviz ile çizdirme
from graphviz import Digraph
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges
def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')

        if n._op:
            #eğer bu değer bir işlemin sonucuysa işlem node'u yaz
            dot.node(name=uid + n._op, label=n._op)
            # ve buna değeri bağla
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        #n1'i n2'nin op node'una bağla
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot

#görev 1
#kendi Value sınıfım. Her yeni Value, kendisini üreten Value'ları ve hangi işlemden çıktığını saklayacak.
class Value:
    def __init__(self, data, _children=(), _op='',label=''):
        self.data = data
        self.grad=0.0
        self._backward = lambda: None #başta hiçbir şey yapmayan fonksiyon ekliyorz. görev3
        self._prev=set(_children)
        self._op=_op
        self.label=label

    def __repr__(self): #ekrana okunaklı şekilde yazdırmak için
        return f"Value(data= {self.data},{self._op})"

    def __add__(self, other): #toplama için fonksiyon
        out= Value(self.data+other.data, (self,other),"+")

        #görev 3 eklemesi
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            """bir node ağaçta birden fazla yere gidiyorsa toplayarak ilerlemeli.
            mesela a=value(3.0). b=a+a. burada a hem self hem other. 
            self.grad += 1.0 * out.grad   # a.grad: 0 → 1
            other.grad += 1.0 * out.grad  # a.grad: 1 → 2 bu kısmı anlamak için yapay zekaya sordum."""
        out._backward = _backward
        return out

    def __mul__(self, other): #çarpma için fonksiyon
        out= Value(self.data*other.data,(self,other),"*")

        #görev 3 eklemesi
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self): #tanh fonksiyonu oluşturuyoruz
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        #görev 3 eklemesi
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
            #total etki= yerel türev * üstten gelen etki
        out._backward = _backward
        return out

    #görev 3 eklemesi
    def backward(self):

        topo = [] #topological sort için boş liste
        visited = set() #tekrar gezmemek için

        #ağacı kökten uca arama
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        #kendinden başlayarak haritayı oluşturma
        build_topo(self)

        #kendi türevini 1 yapıyoruz. L.grad=1 dediğimiz manuel ayarlama
        self.grad = 1.0

        #haritayı terse döndürüyoruz çünkü sondan başlamamız lazım ve hepsi için _backward fonksiyonunu çağırıyoruz
        for node in reversed(topo):
            node._backward()


#daha rahat ayrılsın diye görevleri fonksiyon olarak tanımladım.

def gorev1():
    a= Value(3.0,label="a")
    b= Value(-2.0,label="b")
    c=Value(5.0,label="c")
    e=a*b; e.label = "e"
    d=e+c; d.label = "d"
    print(d)
    print(a*b)
    # Grafiği oluşturmak için:
    grafik = draw_dot(d)
    # Grafiği klasöre 'computation_graph.svg' olarak kaydetip otomatik olarak ekranda açıyor
    grafik.render('grafikler/computation_graph', view=True)


#görev2
#gradienti önce elle hesaplama
def gorev2_1():
    h=0.001 #çok ufak değişiklik yapacağımız değer

    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    e = a * b;
    e.label = "e"
    d = e + c;
    d.label = "d"
    f= Value(-2.0, label="f")
    L= d*f;
    L.label = "L"
    L1=L.data

    a = Value(2.0, label="a")
    b = Value(-3.0+h, label="b") #örnek olarak b değiştiğinde L ne kadar değişir diye baktım ve sonucun f*a olduğunu gördüm.
    #bunu hepsi için bakarak tek tek grad değerlerini hesapladım ve aslında hepsinin türevdeki chain rule olduğunu gördüm.
    c = Value(10.0, label="c")
    e = a * b;
    e.label = "e"
    d = e + c;
    d.label = "d"
    f= Value(-2.0, label="f")
    L= d*f;
    L.label = "L"
    L2=L.data

    L.grad=1
    d.grad=-2
    f.grad=4
    c.grad=-2
    e.grad=-2
    a.grad=-2*-3
    b.grad=-2*2
    print((L2-L1)/h)
    #bunu aslında şöyle düşünebiliriz: formülümüz sigmoid*(a*w+b) idi. f sigmoid veya tanh, d= a*w + b.
    grafik = draw_dot(L)
    # Grafiği klasöre 'computation_graph1.svg' olarak kaydetip otomatik olarak ekranda açıyor
    grafik.render('grafikler/computation_graph2', view=True)


#görev 2'nin ikinci kısmı. yukarıdaki işlemi bir nöron için yapma
def gorev2_2():
    #girdilerimiz
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    #weightler
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    #bias (Karpathy videoda bu değeri yazmıştı)
    b = Value(6.8813735870195432, label='b')

    #çarpıp toplama işlemi
    x1w1 = x1 * w1; x1w1.label = 'x1*w1'
    x2w2 = x2 * w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + b; n.label = 'n' #nöronun bias eklenmiş hali

    #sonucu -1 +1 arasına sıkıştırmak için tanh kullanıyoruz
    o = n.tanh(); o.label = 'o'

    #back propagationa başlıyoruz. en sondan gidiyoruz. o'nun kendisinin grad'ı 1
    o.grad=1
    #sonra bir geride tanh var. onun türevi de 1-tan^2.
    n.grad=1-o.data**2

    #bir geride + işlemi var. bu önceki grad'ı child node'lara dağıtıyor.
    b.grad= 1-o.data**2
    x1w1x2w2.grad=1-o.data**2
    #bunun da gerisinde yine + işlemi var. dağıtıyoruz
    x1w1.grad= 1-o.data**2
    x2w2.grad=1-o.data**2
    #bunun gerisinde * var. çarparak gideceğiz. o yüzden önce x1 w1 x2 w2nin local gradlerine ihtiyacımız var. önceki örnekte de gördüğümüz gibi birbirleri çıkıyor.
    #bu yüzden bu local gradle çarparak hepsinin genel gradini bulmuş oluyoruz.
    x1.grad=w1.data * x1w1.grad
    x2.grad=w2.data * x2w2.grad
    w1.grad=x1.data * x1w1.grad
    w2.grad=x2.data * x2w2.grad

    #grafiği çizdiriyoruz


    grafik = draw_dot(o)
    grafik.render('grafikler/neuron_graph', view=True)

def gorev3():
    #backwardla otomatikleştirme
    #görev 2nin ikinci kısmını otomatikleştirmek için yazdığımız backward fonkiyonunu kullanıyoruz.
    # girdilerimiz
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    # weightler
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    # bias (Karpathy videoda bu değeri yazmıştı)
    b = Value(6.8813735870195432, label='b')

    # çarpıp toplama işlemi
    x1w1 = x1 * w1;
    x1w1.label = 'x1*w1'
    x2w2 = x2 * w2;
    x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2;
    x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + b;
    n.label = 'n'  # nöronun bias eklenmiş hali

    # sonucu -1 +1 arasına sıkıştırmak için tanh kullanıyoruz
    o = n.tanh();
    o.label = 'o'
    #buraya kadar görev 2.2nin aynısıydı. şimdi elle yazmak yerine backward fonksiyonunu çağırıyoruz.
    o.backward()
    grafik = draw_dot(o)
    grafik.render('grafikler/neuron_graph2', view=True)





"""gorev1()
gorev2_1()
gorev2_2()"""
gorev3()





