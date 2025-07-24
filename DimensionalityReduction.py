import numpy as np
import numpy as np
import scipy.linalg
from scipy.linalg import qr, eig, hessenberg

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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

class AnaliseFatorial:

    def __init__(self,X,d,tipo_Matriz='Gram'):
        self.X = X # Matriz recebida pelo usuário
        self.d = d # Nova redução a qual será reduzida X

        self.x_barra = np.mean(self.X, axis=0)  # Calcula a média de cada coluna (dimensão), axis = 1 é para linhas, axis = 0 é para colunas
        self.X_centrada = self.X - self.x_barra  # Centraliza a matriz X subtraindo a média de cada coluna
        

        if tipo_Matriz not in ['Gram', 'Covariancia']:
            raise ValueError("tipo_Matriz deve ser 'Gram' ou 'Covariancia'.")
        
        self.tipo_Matriz = tipo_Matriz

        if self.tipo_Matriz == 'Covariancia':
        
            self.S2 = self.MatrizCovariancia() # Calcula a matriz de covariância S2
            print("Size da matriz de covariância S2:", self.S2.shape)
            self.TeoremaEspectral = EigenvalueCalculator(self.S2)  # Inicializa o Teorema Espectral com a matriz de covariância S2

        elif self.tipo_Matriz == 'Gram':
            self.A = np.transpose(X) @ X  # Matriz de covariância A = X**(t) * X
            self.TeoremaEspectral = EigenvalueCalculator(self.A)

        self.autovalores = np.sort(self.TeoremaEspectral.qr_algorithm(), kind='quicksort')[::-1]  # Ordena os autovalores em ordem decrescente
        self.autovetores = self.TeoremaEspectral.compute_eigenvectors(self.autovalores)
        self.TauQ = self.TauQ()  # Calcula a proporção de variância explicada

        self.Q = self.NormalizaAutovetores()  # Normaliza os autovetores
        self.autovaloresDiagonais = np.diag(self.autovalores)  # Cria a matriz diagonal de autovalores
        self.decomposicaoEspectral = self.Q@self.autovaloresDiagonais@self.Q.T  # Decomposição espectral da matriz A
        
    
    def TauQ(self):
        """ Definem qtd de autovalores a serem considerados na redução """

        if self.d > self.autovalores.shape[0]:
            raise ValueError("d não pode ser maior que o número de autovalores disponíveis.")
        
        return np.sum(self.autovalores[:self.d]) / np.sum(self.autovalores)
    
    def MatrizCovariancia(self):
        """ Calcula a matriz de covariância S2 
        S=(1/(n-1))X_centrado(transposto)*X_centrado​"""

        n = self.X_centrada.shape[0]
        self.S = (1 / (n - 1)) * np.dot(self.X_centrada.T, self.X_centrada)
        return self.S  # Retorna a matriz de covariância S2
        
    def NormalizaAutovetores(self):
        """ Normaliza os autovetores para que tenham norma unitária """
        
        for i in range(self.autovetores.shape[1]):
            self.autovetores[:, i] /= np.linalg.norm(self.autovetores[:, i])
        
        return self.autovetores  # Retorna os autovetores normalizados

    def Reduzida(self):
        """ Retorna a matriz reduzida de dimensão d """
        
        if self.d > self.autovetores.shape[1]:
            raise ValueError("d não pode ser maior que o número de autovetores disponíveis.")
        
        return self.X_centrada @ self.Q[:, :self.d]  # Usa os autovetores normalizados
    
    def run(self):
        """ Executa a análise fatorial e retorna a matriz reduzida """
        
        self.MatrizCovariancia()
        self.NormalizaAutovetores()
        return self.Reduzida()
    
class PCA:    
    def __init__(self,X,d):
        self.X = X # Matriz recebida pelo usuário
        self.d = d # Nova redução a qual será reduzida X
        self.S2 = self.calc_si2() #Matriz Covariâncias amostrais
        self.teoremaEspectral = EigenvalueCalculator(self.S2)
        self.autovalores = np.sort(self.teoremaEspectral.qr_algorithm(), kind='quicksort')[::-1]  # Ordena os autovalores em ordem decrescente
        self.autovetores = self.teoremaEspectral.compute_eigenvectors(self.autovalores)  # Calcula os autovetores
        self.x_barra = np.mean(self.X, axis=0)  # Calcula a média da matriz X
        self.Y = None

    def calc_si2(self):
        """ Calcula Diagonal de covariâncias amostrais 
        
        si2 = (1/(n-1)) * Σ(xij - x̄j)² -- Diagonal principal da matriz de covariância S2,
        onde xij é o valor da j-ésima variável na i-ésima observação e x̄j é a média da j-ésima coluna (variável).

        si2 = (1/(n-1)) * Σ(xki-x̄j)*(xkj-x̄j) -- Outras posições da matriz de covariância S2, fora da diagonal principal.
        Onde xki é o valor da k-ésima variável na i-ésima observação e xkj é o valor da k-ésima variável na j-ésima observação
        n é o número de observações (linhas) e j é o índice da coluna
        """

        n = self.X.shape[0]
        self.S2 = np.zeros((self.X.shape[1], self.X.shape[1]))  # Inicializa a matriz de covariância S2
        for i in range(self.X.shape[1]):
            for j in range(self.X.shape[1]):
                if i == j:
                    self.S2[i, j] = np.sum((self.X[:, i] - np.mean(self.X[:, i])) ** 2) / (n - 1)
                else:
                    self.S2[i, j] = np.sum((self.X[:, i] - np.mean(self.X[:, i])) * (self.X[:, j] - np.mean(self.X[:, j]))) / (n - 1)

        return self.S2  # Retorna a matriz de covariância S2
    
    def TauQ(self):
        """ Definem qtd de autovalores a serem considerados na redução """

        if self.d > self.autovalores.shape[0]:
            raise ValueError("d não pode ser maior que o número de autovalores disponíveis.")
        
        return np.sum(self.autovalores[:self.d]) / np.sum(self.autovalores)
    
    def NormalizaAutovetores(self):
        """ Normaliza os autovetores para que tenham norma unitária """
        
        for i in range(self.autovetores.shape[1]):
            self.autovetores[:, i] /= np.linalg.norm(self.autovetores[:, i])
        
        return self.autovetores  # Retorna os autovetores normalizados

    def run(self):
        """Executa a Análise de Componentes Principais (PCA) e retorna a matriz reduzida Y.
        
        Passos:
        1. Centraliza os dados subtraindo a média (x_barra)
        2. Normaliza os autovetores (se ainda não estiverem normalizados)
        3. Projeta os dados nos primeiros d autovetores (componentes principais)
        
        Retorno:
        Y: Matriz reduzida de dimensão (n, d), onde n é o número de observações
        e d é a dimensão escolhida para a redução.
        """
        # 1. Centralizar os dados
        X_centrada = self.X - self.x_barra
        
        # 2. Garantir que os autovetores estão normalizados
        autovetores_normalizados = self.NormalizaAutovetores()

        print("Representatividade dos d-autovetores de maiorr grau:  ",self.TauQ())
        
        # 3. Selecionar os d primeiros autovetores (componentes principais)
        componentes_principais = autovetores_normalizados[:, :self.d]
        

        # 4. Projetar os dados nos componentes principais (Y = X_centrada * W)
        self.Y = X_centrada @ componentes_principais
        
        return self.Y
   
class analiseInfo:

    def __init__(self,X):
        self.X = X # cada linha é uma observação, cada coluna é uma coordenada do espaço
        self.n = X.shape[0]  # Número de observações
        self.p = X.shape[1]  # Número de variáveis

    def plota_vetores_redimensionados(self):
        if self.p < 2:
            raise ValueError("A matriz X deve ter pelo menos 2 colunas para plotar vetores redimensionados.")
        if self.n < 2:
            raise ValueError("A matriz X deve ter pelo menos 2 linhas para plotar vetores redimensionados.")

        # Cores distintas para cada vetor (usando um colormap)
        cores = plt.cm.get_cmap('hsv', self.n)  # Escolha um colormap (e.g., 'hsv', 'viridis', 'tab20')

        if self.p == 2:
            plt.figure(figsize=(8, 6))
            for i in range(self.n):
                plt.quiver(0, 0, self.X[i, 0], self.X[i, 1], 
                           angles='xy', scale_units='xy', scale=1, 
                           color=cores(i), label=f'Vetor {i+1}')
            plt.xlim(-3.5, 3.5)
            plt.ylim(-3.5, 3.5)
            plt.grid()
            plt.title("Vetores Redimensionados em 2D")
            plt.xlabel("Eixo X")
            plt.ylabel("Eixo Y")
            plt.legend(loc='best', bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
            plt.show()

        elif self.p == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            for i in range(self.n):
                ax.quiver(0, 0, 0, 
                          self.X[i, 0], self.X[i, 1], self.X[i, 2], 
                          color=cores(i), length=1.0, arrow_length_ratio=0.1, 
                          label=f'Vetor {i+1}')
            ax.set_xlim([-3.5, 3.5])
            ax.set_ylim([-3.5, 3.5])
            ax.set_zlim([-3.5, 3.5])
            ax.set_title("Vetores Redimensionados em 3D")
            ax.set_xlabel("Eixo X")
            ax.set_ylabel("Eixo Y")
            ax.set_zlabel("Eixo Z")
            ax.legend(loc='best', bbox_to_anchor=(1.1, 1))
            plt.tight_layout()
            plt.show()

        else:
            raise ValueError("A matriz X deve ter exatamente 2 ou 3 colunas para plotar vetores redimensionados.")


    

if __name__ == "__main__":

    # Definindo a matriz X como uma lista NumPy
    X = np.array([
        [1, 2, -1, 3, 0, 2, 1],
        [2, 4, 1, 1, 2, 2, 2],
        [1, 3, -2, 0, 1, 0, 2],
        [1, 4, -1, 7, 4, 4, 2],
        [1, -1, 3, 3, 1, 3, 2],
        [1, 3, 2, 1, 1, 2, 2],
        [1, 4, -1, 7, 2, 5, 0],
        [1, -1, 3, 3, 3, 2, 2],
        [2, -2, 2, 4, 5, 4, 2]
    ])

  
    
    # Af = AnaliseFatorial(X, 3,tipo_Matriz="Covariancia")  # Exemplo de redução para 3 dimensões
    # # print("Autovalores:", Af.autovalores)
    # # print("Autovetores:", Af.autovetores)
    
    # matriz_reduzida = Af.run()  # Executa a análise fatorial e obtém a matriz reduzida
    # print("TauQ:", Af.TauQ)  # Proporção de variância explicada
    # print("Matriz reduzida:\n", matriz_reduzida)

    # plotador = analiseInfo(matriz_reduzida)  # Cria uma instância da classe analiseInfo com a matriz reduzida
    # plotador.plota_vetores_redimensionados()  # Plota os vetores

    pca = PCA(X,3)
    matriz_reduzida = pca.run()
    print("Matriz reduzida:\n", matriz_reduzida)

    plotador = analiseInfo(matriz_reduzida)  # Cria uma instância da classe analiseInfo com a matriz reduzida
    plotador.plota_vetores_redimensionados()  # Plota os vetores


    
    