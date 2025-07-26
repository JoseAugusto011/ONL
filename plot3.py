import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definindo a função f(x1, x2)
def f(x1, x2):
    return (x1**2)/2 + (3/2)*x2**2 + np.sqrt(7)*x1*x2 + np.sqrt(3)*x1 + np.sqrt(13)

# Criando a grade de pontos para x1 e x2
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

# Ponto crítico calculado
x1_critico = (3 * np.sqrt(3)) / 4
x2_critico = -np.sqrt(21) / 4
z_critico = f(x1_critico, x2_critico)

# Plot 3D
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
ax.scatter(x1_critico, x2_critico, z_critico, color='red', s=100, label='Ponto Crítico')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1, x_2)$')
ax.set_title('Superfície da Função $f(x_1, x_2)$')
ax.legend()

# Curvas de nível
ax2 = fig.add_subplot(122)
contour = ax2.contour(X1, X2, Z, levels=20, cmap='viridis')
ax2.scatter(x1_critico, x2_critico, color='red', s=100, label='Ponto Crítico')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_title('Curvas de Nível de $f(x_1, x_2)$')
ax2.legend()
plt.colorbar(contour, ax=ax2)

plt.tight_layout()
plt.show()