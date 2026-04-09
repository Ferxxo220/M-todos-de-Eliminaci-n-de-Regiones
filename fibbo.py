import math
import matplotlib.pyplot as plt
import pandas as pd
from abc import ABC, abstractmethod

class Optimizador(ABC):
    def __init__(self):
        self.evaluaciones = 0
        self.historial = []

    def f_eval(self, f, x):
        self.evaluaciones += 1
        return f(x)

    @abstractmethod
    def optimizar(self, f, a, b, eps):
        pass

class IntervalHalving(Optimizador):
    def optimizar(self, f, a, b, eps):
        self.evaluaciones = 0
        self.historial = []
        while (b - a) > eps:
            L = b - a
            xm = (a + b) / 2
            x1 = a + L / 4
            x2 = b - L / 4
            fx1 = self.f_eval(f, x1)
            fxm = self.f_eval(f, xm)
            fx2 = self.f_eval(f, x2)
            self.historial.append(xm)
            if fx1 < fxm:
                b = xm
            elif fx2 < fxm:
                a = xm
            else:
                a = x1
                b = x2
        return (a + b) / 2

class FibonacciSearch(Optimizador):
    def optimizar(self, f, a, b, eps):
        self.evaluaciones = 0
        self.historial = []
        L = b - a
        fib = [1, 1]
        while fib[-1] < (L / eps):
            fib.append(fib[-1] + fib[-2])
        fib.append(fib[-1] + fib[-2])
        n = len(fib) - 2
        for k in range(2, n + 2):
            lk = (fib[n - k + 2] / fib[n + 1]) * L
            x1 = a + lk
            x2 = b - lk
            fx1 = self.f_eval(f, x1)
            fx2 = self.f_eval(f, x2)
            self.historial.append((a + b) / 2)
            if fx1 < fx2:
                b = x2
            else:
                a = x1
        return (a + b) / 2

class GoldenSection(Optimizador):
    def optimizar(self, f, a, b, eps):
        self.evaluaciones = 0
        self.historial = []
        phi = (math.sqrt(5) - 1) / 2
        L = b - a
        x1 = a + (1 - phi) * L
        x2 = a + phi * L
        fx1 = self.f_eval(f, x1)
        fx2 = self.f_eval(f, x2)
        while (b - a) > eps:
            self.historial.append((a + b) / 2)
            if fx1 < fx2:
                b = x2
                x2 = x1
                fx2 = fx1
                L = b - a
                x1 = a + (1 - phi) * L
                fx1 = self.f_eval(f, x1)
            else:
                a = x1
                x1 = x2
                fx1 = fx2
                L = b - a
                x2 = a + phi * L
                fx2 = self.f_eval(f, x2)
        return (a + b) / 2

def ejecutar_pruebas():
    funciones = [
        {"f": lambda x: x**2 + 54/x if x > 0 else 1e9, "lims": [0.1, 10], "id": "x2 + 54/x"},
        {"f": lambda x: x**4 + x**2 - 33, "lims": [-2.5, 2.5], "id": "x4 + x2 - 33"},
        {"f": lambda x: 3*x**4 - 8*x**3 - 6*x**2 + 12*x, "lims": [-1.5, 3], "id": "Polinomial_3"}
    ]
    precisiones = [0.5, 0.1, 0.01, 0.0001]
    metodos = {
        "Interval Halving": IntervalHalving(),
        "Fibonacci": FibonacciSearch(),
        "Golden Section": GoldenSection()
    }
    data = []
    for fn in funciones:
        plt.figure(figsize=(8, 5))
        for nombre, met in metodos.items():
            for eps in precisiones:
                x_opt = met.optimizar(fn["f"], fn["lims"][0], fn["lims"][1], eps)
                data.append([fn["id"], nombre, eps, met.evaluaciones, round(x_opt, 6)])
            plt.plot(met.historial, label=nombre)
        plt.title(f"Convergencia: {fn['id']}")
        plt.legend()
        plt.grid(True)
        plt.show()
    df = pd.DataFrame(data, columns=["Funcion", "Metodo", "Precision", "Eval", "x_opt"])
    print(df.to_string(index=False))

if __name__ == "__main__":
    ejecutar_pruebas()