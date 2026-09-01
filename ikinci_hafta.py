import random
import math
import torch #görev 4 eklemesi
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
        other=other if isinstance(other,Value) else Value(other) #görev 4 eklemesi. Value objesi değilse onu Value'ya sarıyor.
        out= Value(self.data+other.data, (self,other),"+")

        #görev 3 eklemesi
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            """bir node ağaçta birden fazla yere gidiyorsa toplayarak ilerlemeli.
            videoda da bu örneği verip bu kısmı düzeltmişti.
            mesela a=value(3.0). b=a+a. burada a hem self hem other. 
            self.grad += 1.0 * out.grad   # a.grad: 0 → 1
            other.grad += 1.0 * out.grad  # a.grad: 1 → 2 bu kısmı anlamak için yapay zekaya sordum."""
        out._backward = _backward
        return out


    def __mul__(self, other): #çarpma için fonksiyon
        other = other if isinstance(other, Value) else Value(other)
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

    def __pow__(self, other): #görev 4 eklemesi: üs için
        assert isinstance(other, (int, float)), "sadece int/float üsler destekleniyor"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad #türevinde üs başa çarpım olarak gelir ve bir azalır.
        out._backward = _backward
        return out

    def __radd__(self,other): #görev 4 eklemesi: sıraları değişikse yerlerini değiştirip tekrar toplamaya çalışyıor.
        return self+other

    def __rmul__(self,other): #görev 4 eklemesi: sıraları değişikse yerlerini değiştirip tekrar çarpmaya çalışıyor.
        return self*other

    def __truediv__(self, other): #görev 4 eklemesi: bölme işlemi. yani aslında ikincinin -1. kuvvetiyle çarpma
        return self* other**-1

    def __sub__(self, other): #görev 4 eklemesi: çıkarma işlemi. aslında ikincinin işaretini negatif yapıp toplama
        return self+(-other)

    def __neg__(self): #görev 4 eklemesi: işaret değiştirme
        return self*(-1)

    def exp(self): #görev 4 eklemesi
        x = self.data
        out= Value(math.exp(x),(self,),"exp")
        def _backward():
            self.grad+=out.data * out.grad #e^x in türevi e^x. onu da out'ta hesapladığımız için out.data*out.grad diyoruz.
        out._backward = _backward
        return out

#daha rahat ayrılsın diye görevleri fonksiyon olarak tanımladım.


#Value sınıfını yazdım. hangi işlemden çıktıklarını saklıyor.
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

def gorev4():
    #görev 3'teki girdi kısımlarını kopyala yapıştır yapıyoruz. tanh'i aşağıda parçalayacağız.
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


    #görev 4 eklemesi: tanh'ı parçalama
    #tanh= (e**2x -1) / (e**2x +1)
    e= (2*n).exp() #e**2x'i iki kere kullanacağımız için bir intermediate variable oluşturuyoruz.
    o= (e-1) / (e+1)
    o.label = 'o'
    o.backward()
    grafik = draw_dot(o)
    grafik.render('grafikler/neuron_graph3', view=True)



def gorev4_dogrulama(): #backward fonksiyonuyla hesapladığımız gradyanları pytorch kullanarak ve numerical derivative kullanarak da hesaplayıp karşılaştırıyoruz.
    x1 = torch.Tensor([2.0]).double() #benim yazdığım x1=Value(2.0)
    x1.requires_grad = True
    x2 = torch.Tensor([0.0]).double()
    x2.requires_grad = True
    w1 = torch.Tensor([-3.0]).double()
    w1.requires_grad = True
    w2 = torch.Tensor([1.0]).double()
    w2.requires_grad = True
    b = torch.Tensor([6.8813735870195432]).double()
    b.requires_grad = True

    n = x1 * w1 + x2 * w2 + b
    o = torch.tanh(n) #benim yazdığım o=n.tanh()

    print(o.data.item())
    o.backward()

    print('x2', x2.grad.item()) #benim yazdığım print(x2.grad)
    print('w2', w2.grad.item())
    print('x1', x1.grad.item())
    print('w1', w1.grad.item())
    print ("------ PyTorch sonu ------")

    print("Benim yaptığım backward() ile hesaplama")

    mx1= Value(2.0)
    mx2=Value(0.0)
    mw1=Value(-3.0)
    mw2=Value(1.0)
    mb=Value(6.8813735870195432)

    mn= mx1*mw1 + mx2*mw2+mb
    me= (mn*2).exp()
    mo= (me-1)/(me+1)
    mo.backward()

    print(f"o değeri: {mo.data}")
    print(f"x2: {mx2.grad}")
    print(f"w2: {mw2.grad}")
    print(f"x1: {mx1.grad}")
    print(f"w1: {mw1.grad}")
    print("------ Backward fonksiyonuyla hesaplama sonu -------")

    print("Numerical derivative ile hesaplama")

    h=0.0001
    nx1= Value(2.0)
    nx2= Value(0.0)
    nw1= Value(-3.0)
    nw2= Value(1.0)
    nb= Value(6.8813735870195432)
    nn1= nx1*nw1 + nx2*nw2+nb
    ne1=(nn1*2).exp()
    no1= (ne1-1)/(ne1+1)
    L1= no1.data

    nx1h= Value(2.0)
    nx2h= Value(0.0)
    nw1h= Value(-3.0+h)
    nw2h= Value(1.0)
    nbh= Value(6.8813735870195432)
    nn2= nx1h*nw1h + nx2h*nw2h+nbh
    ne2=(nn2*2).exp()
    no2= (ne2-1)/(ne2+1)
    L2= no2.data

    num_grad= (L2-L1)/h

    print(f"w1'in gradyanı: {num_grad}")

#gorev1()
#gorev2_1()
#gorev2_2()
#gorev3()
#gorev4()
#gorev4_dogrulama()


def gorev5():
    class Neuron:

        def __init__(self, nin):  # nin: number of inputs

            self.w = [Value(random.uniform(-1, 1)) for i in
                      range(nin)]  # -1 ile 1 arasında değişen değerlerden oluşan weights listesi

            self.b = Value(random.uniform(-1, 1))  # -1 ile 1 arasında değişen bias değeri

        def __call__(self, x):
            act = self.b

            for wi, xi in zip(self.w, x):
                act += xi * wi
            out = act.tanh()
            return out

        def parameters(self):
            return self.w + [self.b]

    class Layer:

        def __init__(self, nin, nout):  # nout: number of outputs
            self.neurons=[Neuron(nin) for i in range (nout)]

        def __call__(self, x):
            outs=[n(x) for n in self.neurons]
            return outs[0] if len(outs) == 1 else outs

        def parameters(self):
            params=[]
            for n in self.neurons:
                params+=n.parameters() #append liste içinde liste oluşturacağı için onu kullanmadım.
            return params

    class MLP: #multi layer perceptron

        def __init__(self, nin, nouts):
            sz= [nin] + nouts
            self.layers=[Layer(sz[i],sz[i+1]) for i in range (len(nouts))]

        def __call__(self, x):
            for layer in self.layers:
                x=layer(x)
            return x

        def parameters(self):
            params=[]
            for layer in self.layers:
                params+=layer.parameters()
            return params

    """grafik = draw_dot(n(x))
    grafik.render('grafikler/mlpgraph', view=True)"""
    n=MLP(3,[4,4,1])
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0]  # Ağdan bulmasını istediğimiz doğru cevaplar
    n = MLP(3, [4, 4, 1])
    ypred=[n(x) for x in xs]
    loss_sum = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    for ygt, yout in zip (ys, ypred):
        loss= (yout - ygt)**2
        print (loss.data)

    print("Güncel Hata (Loss):", loss_sum.data)

gorev5()












