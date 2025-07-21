import numpy as np
import autograd.numpy as anp
from autograd import jacobian

class FuncRmToRn:
    """
    Classe que representa uma função F: R^m -> R^n e calcula o Jacobiano.
    
    Parâmetros:
    ----------
    func : callable
        Função que recebe um array numpy de shape (m,) e retorna um array de shape (n,).
    """
    
    def __init__(self, func):
        self.func = func
        self._jacobian = jacobian(self.func)  # Autograd calcula o Jacobiano
    
    def __call__(self, x):
        """Avalia a função no ponto x."""
        return np.array(self.func(x))
    
    def jacobian(self, x):
        """
        Retorna a matriz Jacobiana da função no ponto x.
        
        Parâmetros:
        ----------
        x : array_like
            Ponto no qual o Jacobiano será calculado (shape (m,)).
            
        Retorna:
        -------
        array
            Matriz Jacobiana no ponto x (shape (n, m)).
        """
        return np.array(self._jacobian(x))


# Exemplo de uso:
if __name__ == "__main__":
    # Define uma função R^2 -> R^3 (exemplo: F(x, y) = [x + y, x^2, y^3])
    def vector_func(x):
        return anp.array([x[0] + x[1], x[0]**2, x[1]**3])  # Usar autograd.numpy
    
    # Cria a instância da classe
    func = FuncRmToRn(vector_func)
    
    # Ponto para avaliar
    x = np.array([1.0, 2.0])
    
    print("Função:", func(x))
    print("Jacobiano:\n", func.jacobian(x))