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
        self._prev=set(_children)
        self._op=_op
        self.label=label

    def __repr__(self): #ekrana okunaklı şekilde yazdırmak için
        return f"Value(data= {self.data},{self._op})"

    def __add__(self, other): #toplama için fonksiyon
        out= Value(self.data+other.data, (self,other),"+")
        return out

    def __mul__(self, other): #çarpma için fonksiyon
        out= Value(self.data*other.data,(self,other),"*")
        return out

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
grafik.render('computation_graph', view=True)


#görev2
#gradienti önce elle hesaplama
def grad_manuel_hesaplama():
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
    grafik = draw_dot(L)
    # Grafiği klasöre 'computation_graph1.svg' olarak kaydetip otomatik olarak ekranda açıyor
    grafik.render('computation_graph1', view=True)

grad_manuel_hesaplama()




