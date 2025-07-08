import numpy as np
import scipy.linalg
from scipy.linalg import qr, eig, hessenberg

class EigenvalueCalculator:
    def __init__(self, matrix):
        """
        Inicializa a calculadora com uma matriz quadrada.
        
        Args:
            matrix: numpy.ndarray - matriz quadrada para cálculo dos autovalores
        """
        self.A = np.array(matrix, dtype=float)
        assert self.A.shape[0] == self.A.shape[1], "A matriz deve ser quadrada"
        self.n = self.A.shape[0]
    
    def von_mises(self, max_iter=1000, tol=1e-6):
        """
        Iteração de Von Mises para calcular o maior autovalor (em módulo).
        
        Args:
            max_iter: int - número máximo de iterações
            tol: float - tolerância para convergência
            
        Returns:
            (autovalor, autovetor, num_iter)
        """
        x = np.random.rand(self.n)
        x = x / np.linalg.norm(x)
        
        lambda_prev = 0
        for k in range(max_iter):
            x_new = self.A @ x
            lambda_k = np.dot(x, x_new)
            x_new = x_new / np.linalg.norm(x_new)
            
            if np.abs(lambda_k - lambda_prev) < tol:
                break
                
            x = x_new
            lambda_prev = lambda_k
            
        return lambda_k, x, k+1
    
    def inverse_power_method(self, mu=0, max_iter=1000, tol=1e-6):
        """
        Método da potência inversa para calcular o menor autovalor ou o mais próximo de mu.
        
        Args:
            mu: float - deslocamento para encontrar autovalor mais próximo
            max_iter: int - número máximo de iterações
            tol: float - tolerância para convergência
            
        Returns:
            (autovalor, autovetor, num_iter)
        """
        I = np.eye(self.n)
        A_shifted = self.A - mu * I
        LU, piv = scipy.linalg.lu_factor(A_shifted)
        
        x = np.random.rand(self.n)
        x = x / np.linalg.norm(x)
        
        lambda_prev = np.inf
        for k in range(max_iter):
            x_new = scipy.linalg.lu_solve((LU, piv), x)
            lambda_k = np.dot(x, x_new)
            x_new = x_new / np.linalg.norm(x_new)
            
            if np.abs(1/lambda_k - 1/lambda_prev) < tol:
                break
                
            x = x_new
            lambda_prev = lambda_k
            
        return (1/lambda_k) + mu, x, k+1
    
    def deflation(self, lambda1, v1):
        """
        Deflação para reduzir a matriz após encontrar um autovalor/autovetor.
        
        Args:
            lambda1: float - autovalor conhecido
            v1: numpy.ndarray - autovetor correspondente normalizado
            
        Returns:
            numpy.ndarray - matriz reduzida de dimensão (n-1)x(n-1)
        """
        v1 = v1.reshape(-1, 1)
        B = self.A - lambda1 * (v1 @ v1.T)
        return B[:-1, :-1]  # Versão simplificada - pode ser melhorada
    
    def qr_algorithm(self, max_iter=100, tol=1e-6):
        """
        Algoritmo QR para calcular todos os autovalores.
        
        Args:
            max_iter: int - número máximo de iterações
            tol: float - tolerância para convergência
            
        Returns:
            numpy.ndarray - array com os autovalores
        """
        A_k = np.copy(self.A)
        n = self.n
        
        for k in range(max_iter):
            Q, R = qr(A_k)
            A_next = R @ Q
            
            # Verifica convergência na subdiagonal
            off_diag = np.sum(np.abs(np.tril(A_next, -1)))
            if off_diag < tol:
                break
                
            A_k = A_next
            
        return np.diag(A_k)
    
    def jacobi_method(self, max_iter=100, tol=1e-6):
        """
        Método de Jacobi para matrizes simétricas (calcula todos os autovalores).
        
        Args:
            max_iter: int - número máximo de iterações
            tol: float - tolerância para convergência
            
        Returns:
            numpy.ndarray - array com os autovalores
        """
        assert np.allclose(self.A, self.A.T), "Método de Jacobi requer matriz simétrica"
        
        A_k = np.copy(self.A)
        n = self.n
        V = np.eye(n)
        
        for _ in range(max_iter):
            # Encontra o maior elemento fora da diagonal
            off_diag = np.abs(np.tril(A_k, -1))
            p, q = np.unravel_index(np.argmax(off_diag), off_diag.shape)
            
            if off_diag[p, q] < tol:
                break
                
            # Calcula a rotação de Jacobi
            if A_k[p, p] == A_k[q, q]:
                theta = np.pi/4
            else:
                theta = 0.5 * np.arctan(2*A_k[p, q] / (A_k[p, p] - A_k[q, q]))
                
            c = np.cos(theta)
            s = np.sin(theta)
            
            J = np.eye(n)
            J[p, p] = c
            J[q, q] = c
            J[p, q] = s
            J[q, p] = -s
            
            A_k = J.T @ A_k @ J
            V = V @ J
            
        return np.diag(A_k)
    
    def lanczos(self, m=None, tol=1e-6):
        """
        Algoritmo de Lanczos para matrizes simétricas (tridiagonalização).
        
        Args:
            m: int - número de iterações (default: n)
            tol: float - tolerância para ortogonalidade
            
        Returns:
            (T, Q) - matriz tridiagonal T e matriz ortogonal Q
        """
        assert np.allclose(self.A, self.A.T), "Lanczos requer matriz simétrica"
        
        n = self.n
        m = m if m is not None else n
        
        Q = np.zeros((n, m+1))
        T = np.zeros((m, m))
        
        # Vetor inicial aleatório
        q = np.random.rand(n)
        q = q / np.linalg.norm(q)
        Q[:, 0] = q
        
        beta = 0
        for j in range(m):
            # Multiplicação matriz-vetor
            if j == 0:
                r = self.A @ q
            else:
                r = self.A @ q - beta * Q[:, j-1]
                
            alpha = np.dot(q, r)
            r = r - alpha * q
            beta = np.linalg.norm(r)
            
            T[j, j] = alpha
            if j < m-1:
                T[j, j+1] = beta
                T[j+1, j] = beta
                
                if beta < tol:  # Quebra se beta for muito pequeno
                    break
                    
                q_new = r / beta
                Q[:, j+1] = q_new
                q = q_new
                
        return T[:j+1, :j+1], Q[:, :j+1]
    
    def arnoldi(self, m=None, tol=1e-6):
        """
        Algoritmo de Arnoldi para matrizes não simétricas (forma de Hessenberg).
        
        Args:
            m: int - número de iterações (default: n)
            tol: float - tolerância para ortogonalidade
            
        Returns:
            (H, Q) - matriz de Hessenberg H e matriz ortogonal Q
        """
        n = self.n
        m = m if m is not None else n
        
        Q = np.zeros((n, m+1))
        H = np.zeros((m+1, m))
        
        # Vetor inicial aleatório
        q = np.random.rand(n)
        q = q / np.linalg.norm(q)
        Q[:, 0] = q
        
        for j in range(m):
            # Multiplicação matriz-vetor
            v = self.A @ Q[:, j]
            
            # Ortogonalização de Gram-Schmidt modificada
            for i in range(j+1):
                H[i, j] = np.dot(Q[:, i], v)
                v = v - H[i, j] * Q[:, i]
                
            H[j+1, j] = np.linalg.norm(v)
            
            if H[j+1, j] < tol:  # Quebra se norma for muito pequena
                break
                
            Q[:, j+1] = v / H[j+1, j]
            
        return H[:j+1, :j+1], Q[:, :j+1]
    
    def get_all_eigenvalues(self, method='qr', **kwargs):
        """
        Calcula todos os autovalores usando um método especificado.
        
        Args:
            method: str - 'qr', 'jacobi', 'lanczos' ou 'arnoldi'
            **kwargs: argumentos adicionais para os métodos
            
        Returns:
            numpy.ndarray - array com os autovalores
        """
        if method == 'qr':
            return self.qr_algorithm(**kwargs)
        elif method == 'jacobi':
            return self.jacobi_method(**kwargs)
        elif method == 'lanczos':
            T, _ = self.lanczos(**kwargs)
            return np.linalg.eigvals(T)
        elif method == 'arnoldi':
            H, _ = self.arnoldi(**kwargs)
            return np.linalg.eigvals(H)
        else:
            raise ValueError(f"Método desconhecido: {method}")

    def compute_eigenvectors(self, eigenvalues, max_iter=1000, tol=1e-6):
        """
        Calcula os autovetores associados a autovalores conhecidos usando o método da potência inversa com deslocamento.
        
        Args:
            eigenvalues: numpy.ndarray - array com os autovalores já calculados.
            max_iter: int - número máximo de iterações por autovalor.
            tol: float - tolerância para convergência.
            
        Returns:
            numpy.ndarray - matriz onde cada coluna é um autovetor normalizado.
        """
        n = self.n
        eigenvectors = np.zeros((n, n))
        
        for i, lambda_k in enumerate(eigenvalues):
            # Usa o método da potência inversa com deslocamento lambda_k
            _, v, _ = self.inverse_power_method(mu=lambda_k, max_iter=max_iter, tol=tol)
            eigenvectors[:, i] = v
        
        return eigenvectors

    def sort_eigenvalues(self, eigenvalues):
        """
        Ordena os autovalores em ordem crescente. Usa algoritmo Quicksort do NumPy.
        
        Args:
            eigenvalues: numpy.ndarray - array com os autovalores a serem ordenados.
            
        Returns:
            numpy.ndarray - autovalores ordenados.
        """
        return np.sort(eigenvalues, kind='quicksort')[::-1]  # Ordena em ordem decrescente

    def return_mn_matrix(self,V,A):
        """
        Retorna a matriz original.

        Args:
            V: numpy.ndarray - matriz de autovetores.
            A: numpy.ndarray - matriz de autovalores (diagonal).

        Procs:

            A=VΛV−1

        onde Λ é a matriz diagonal de autovalores.



        
        Returns:
            numpy.ndarray - matriz original.
        """
        # return V @ np.diag(A) @ np.linalg.inv(V)
        return V @ np.diag(A) @ V.T


    def print_reducted_matrix(self, V,m):
        """
        Imprime a matriz reduzida.
        
        Args:
            V: numpy.ndarray - matriz de autovetores.
            m: int - número de autovetores a serem considerados.
        """
        if m > V.shape[1]:
            raise ValueError("m não pode ser maior que o número de autovetores disponíveis.")
        
        print("Matriz reduzida (considerando os primeiros m autovetores):")
        print(V[:, :m])

    



if __name__ == "__main__":
    # Matriz de exemplo simétrica
    A_sym = np.array([[4, 1, 1],
                      [1, 3, 2],
                      [1, 2, 5]])
    
    calc_sym = EigenvalueCalculator(A_sym)
    
    # Passo 1: Calcular os autovalores (usando QR, Jacobi, etc.)
    eigenvalues = calc_sym.qr_algorithm()
    print("Autovalores calculados:", eigenvalues)

    # # Ordenar os autovalores
    # eigenvalues = calc_sym.sort_eigenvalues(eigenvalues)
    # print("Autovalores ordenados:", eigenvalues)
    
    # Passo 2: Calcular os autovetores associados
    eigenvectors = calc_sym.compute_eigenvectors(eigenvalues)
    print("Autovetores (colunas) Associado ao n° - ésimo maior autovalor :\n", eigenvectors)
    
    # Verificação: Aplicar A * v - lambda * v (deve ser próximo de zero)
    for i in range(len(eigenvalues)):
        Av = A_sym @ eigenvectors[:, i]
        lambda_v = eigenvalues[i] * eigenvectors[:, i]
        residual = np.linalg.norm(Av - lambda_v)
        print(f"Resíduo para autovalor {eigenvalues[i]:.6f}: {residual:.2e}")


    # Matriz reduzida
    m = 2  # Número de autovetores a considerar
    calc_sym.print_reducted_matrix(eigenvectors, m)

    # Matriz original reconstruída
    reconstructed_matrix = calc_sym.return_mn_matrix(eigenvectors, eigenvalues)

    print("Matriz original reconstruída:\n", reconstructed_matrix)

    # Verificação de reconstrução

    residual_reconstruction = np.linalg.norm(A_sym - reconstructed_matrix)
    print(f"Resíduo da reconstrução da matriz original: {residual_reconstruction:.2e}")

