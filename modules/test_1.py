import numpy as np
import matplotlib.pyplot as plt

# Параметры магнита
μ0 = 4 * np.pi * 1e-7  # Магнитная постоянная (Гн/м)
Br = 1.2  # Остаточная индукция магнита (Тл)
D = 0.0195  # Диаметр магнита (м)
R = D / 2  # Радиус магнита (м)
L = 0.01  # Половина длины магнита (м)

# Функция для силы между двумя магнитами на расстоянии d
def force_between_magnets(d):
    term1 = (d + L) / ((d + L)**2 + R**2)**(1.5)
    term2 = (d - L) / ((d - L)**2 + R**2)**(1.5)
    return (np.pi * R**2 * Br**2 / μ0) * (term1 - term2)

# Массив значений расстояния d от 0.01 м до 0.1 м
d_values = np.linspace(0.005, 0.04, 500)
force_values = force_between_magnets(d_values)

# Построение графика
plt.plot(d_values, force_values, label=r'$F(d)$')
plt.title(r'Сила отталкивания между двумя магнитами вдоль оси')
plt.xlabel('Расстояние между магнитами d (м)')
plt.ylabel('Сила F (Н)')
plt.grid(True)
plt.legend()
plt.show()
