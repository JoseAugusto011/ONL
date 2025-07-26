import numpy as np
from dfx import FuncRn
from scipy.optimize import minimize_scalar  # Para minimizar ϕ(t) numericamente

class GradienteDescendente:

    def __init__(self, f_x, x_barra,tipo_gradiente="numeric"):
        self.f_x = f_x  # Função objetivo
        self.x_barra = x_barra  # Ponto inicial
        self.d = None  # Direção de descida
        self.t = None  # Tamanho do passo
        self.tipo_gradiente = tipo_gradiente
        if self.tipo_gradiente == "analytic":
            self.dfx = FuncRn(self.f_x)

    def calcD(self):
        """Calcula a direção de descida (gradiente negativo normalizado)."""

        if self.tipo_gradiente == "analytic":
            grad = -self.dfx.gradient(self.x_barra)
        if not self.tipo_gradiente == "analytic":
            grad = self._gradiente_numerico(self.x_barra)

        self.d = -grad / np.linalg.norm(grad)  # Normaliza a direção
        
        

    def _gradiente_numerico(self, x, h=1e-5):
        """Calcula o gradiente numericamente (caso não tenha uma fórmula analítica)."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            grad[i] = (self.f_x(x_plus) - self.f_x(x_minus)) / (2 * h)
        return grad

    def buscaExata(self):
        """Calcula o passo ótimo t usando busca exata (minimizando ϕ(t))."""
        def phi(t):
            return self.f_x(self.x_barra + t * self.d)

        # Minimiza ϕ(t) usando um método numérico (Brent)
        result = minimize_scalar(phi, bounds=(0, 1), method='bounded')
        self.t = result.x

    def atualizar_x(self):
        """Atualiza x_barra para o próximo passo."""
        self.x_barra = self.x_barra + self.t * self.d

    def otimizar(self, max_iter=1000, tol=1e-6):


        for _ in range(max_iter):


            self.calcD()
            self.buscaExata()
            self.atualizar_x()
            
            if not self.tipo_gradiente == "analytic":
                grad = self._gradiente_numerico(self.x_barra)
            #ou 
            if self.tipo_gradiente == "analytic":
                grad = self.dfx.gradient(self.x_barra)

            self.printValues(grad)

            if np.linalg.norm(grad) < tol: # Parada
                break

    def printValues(self,grad):
        print("\n\n\t-----------Relatório-----------")
        print("x_barra: ", self.x_barra)
        print("t: ", self.t)
        print("d: ", self.d)
        print("grad: ", grad)
        print("f_x(x_barra): ", self.f_x(self.x_barra))
        print("Novo Ponto  ",self.x_barra+self.t*self.d)
        print("\n\n")

if __name__ == "__main__":
    # Exemplo: Função quadrática f(x) = (x_0 - 2)^2 + (x_1 - 3)^2
    f_x = lambda x: (x[0] - 2)**2 + (x[1] - 3)**2
    x_barra = np.array([0.0, 0.0])  # Ponto inicial

    gd = GradienteDescendente(f_x, x_barra,"analytic")
    gd.otimizar()
    # gd.calcD()  # Calcula a direção de descida
    # gd.buscaExata()  # Calcula o passo ótimo t

    # print(f"Direção de descida (d): {gd.d}")
    # print(f"Tamanho do passo ótimo (t): {gd.t}")
    # print(f"Novo ponto (x + t*d): {gd.x_barra + gd.t * gd.d}")