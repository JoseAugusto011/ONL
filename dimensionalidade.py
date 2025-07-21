import numpy as np

class DimensionalityReduction:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.x_barra = None
        self.G = None
        self.L = None
    
    def fit(self, X):
        # Estimadores de máxima verossimilhança
        self.x_barra = np.mean(X, axis=0)
        X_centered = X - self.x_barra
        n = X.shape[0]
        S = (X_centered.T @ X_centered) / n  # Matriz de covariância (ML)
        
        # Decomposição espectral: S = G^T L G
        autovalores, autovetores = np.linalg.eigh(S)
        idx = np.argsort(autovalores)[::-1]  # Ordena em ordem decrescente
        self.L = np.diag(autovalores[idx])
        self.G = autovetores[:, idx]
        
        if self.n_components is None:
            self.n_components = X.shape[1]
    
    def transform(self, X):
        X_centered = X - self.x_barra
        Y = X_centered @ self.G  # Componentes principais
        return Y[:, :self.n_components]  # Primeiras r componentes
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Exemplo de uso:
if __name__ == "__main__":
    # Dados de exemplo
    X = np.random.rand(100, 200)  # 100 amostras, 5 features
    
    # Redução para 2 componentes
    pca = DimensionalityReduction(n_components=2)
    Y = pca.fit_transform(X)
    print("Dados originais shape:", X.shape)
    print("Dados reduzidos shape:", Y.shape)