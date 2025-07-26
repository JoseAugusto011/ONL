import numpy as np
import matplotlib.pyplot as plt

# Definir el rango de valores para x y y
x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.linspace(-2 * np.pi, 2 * np.pi, 100)
X, Y = np.meshgrid(x, y)

# Calcular los valores de f(x, y)
Z = np.cos(X) + np.cos(Y)

# Crear el gráfico de contornos
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(contour)

# Etiquetas
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mapa de Contornos de f(x, y) = cos(x) + cos(y)')

# Mostrar el gráfico
plt.show()
