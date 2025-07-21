import numpy as np
import autograd.numpy as anp  # Autograd usa numpy modificado
from autograd import grad, elementwise_grad
from autograd import hessian

class FuncRn:
    """
    Classe que representa uma função f: R^n -> R e calcula gradiente e Hessiana.
    
    Parâmetros:
    ----------
    func : callable
        Função que recebe um array numpy de shape (n,) e retorna um escalar.
    """
    
    def __init__(self, func):
        self.func = func
        
        # Cria as funções para gradiente e Hessiana usando autograd
        self._gradient = grad(self.func)
        self._hessian = hessian(self.func)
    
    def __call__(self, x):
        """ Avalia a função no ponto x. """
        return self.func(x)
    
    def gradient(self, x):
        """
        Retorna o gradiente da função no ponto x.
        
        Parâmetros:
        ----------
        x : array_like
            Ponto no qual o gradiente será calculado (shape (n,)).
            
        Retorna:
        -------
        array
            Vetor gradiente no ponto x (shape (n,)).
        """
        return np.array(self._gradient(x))
    
    def hessian(self, x):
        """
        Retorna a matriz Hessiana da função no ponto x.
        
        Parâmetros:
        ----------
        x : array_like
            Ponto no qual a Hessiana será calculada (shape (n,)).
            
        Retorna:
        -------
        array
            Matriz Hessiana no ponto x (shape (n, n)).
        """
        return np.array(self._hessian(x))


# Exemplo de uso:
if __name__ == "__main__":
    # Define uma função R^n -> R (exemplo: f(x, y) = x^2 + y^2)
    def quadratica(x):
        return anp.sum(x**2)  # Nota: usar autograd.numpy (anp) para operações
    
    # Cria a instância da classe
    func = FuncRn(quadratica)
    
    # Ponto para avaliar
    x = np.array([1.0, 2.0])
    
    print("Função:", func(x))
    print("Gradiente:", func.gradient(x))
    print("Hessiana:", func.hessian(x))