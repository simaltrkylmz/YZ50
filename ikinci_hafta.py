#görev 1
#kendi Value sınıfım. Her yeni Value, kendisini üreten Value'ları ve hangi işlemden çıktığını saklayacak.
class Value:
    def __init__(self, data):
        self.data = data

    def __repr__(self): #ekrana okunaklı şekilde yazdırmak için
        return f"Value(data= {self.data})"

    def __add__(self, other): #toplama için fonksiyon
        out= Value(self.data+other.data)
        return out

    def __mul__(self, other): #çarpma için fonksiyon
        out= Value(self.data*other.data)
        return out

a= Value(3.0)
b= Value(-2.0)
c=a+b
print(c)
print(c.data)
print(a*b)
